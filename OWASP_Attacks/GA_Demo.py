import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from copy import deepcopy

st.title("Text-Based Adversarial Attack Demo (FGSM vs PGD vs Token-GCG on BERT)")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# --- Helper functions ---

def get_target_label(target_class):
    return 1 if target_class == "Positive" else 0


def get_pred_label(idx):
    return "Positive" if idx == 1 else "Negative"

def get_confidence(pred, idx):
    return float(pred[0, idx].item())

def highlight_tokens(orig_tokens, mod_tokens):
    # Returns HTML with <mark> for changed tokens
    html = []
    for o, m in zip(orig_tokens, mod_tokens):
        if o == m:
            html.append(m)
        else:
            html.append(f"<mark>{m}</mark>")
    return " ".join(html)

def get_synonym(token):
    # Placeholder: just returns token for now, or [MASK] for demo
    # You can expand with a real synonym dictionary or API
    return "[MASK]"

# --- Attack functions ---

def fgsm_attack(model, embeddings, attention_mask, target_label, epsilon=0.1):
    adv_embeddings = embeddings.clone().detach().requires_grad_(True)
    outputs = model(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
    logits = outputs.logits
    # For targeted attack, minimize the loss for the target class
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label], device=logits.device))
    model.zero_grad()
    if adv_embeddings.grad is not None:
        adv_embeddings.grad.zero_()
    loss.backward()
    grad = adv_embeddings.grad.data
    # For targeted attack, move embeddings toward target class (negative gradient)
    adv_embeddings = adv_embeddings - epsilon * grad.sign()
    adv_embeddings = torch.clamp(adv_embeddings, embeddings - epsilon, embeddings + epsilon)
    return adv_embeddings.detach()

def pgd_attack(model, embeddings, attention_mask, target_label, epsilon=0.1, alpha=0.01, num_iter=40):
    orig_embeddings = embeddings.clone().detach()
    adv_embeddings = embeddings.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        outputs = model(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label], device=logits.device))
        model.zero_grad()
        if adv_embeddings.grad is not None:
            adv_embeddings.grad.zero_()
        loss.backward()
        grad = adv_embeddings.grad.data
        # For targeted attack, move embeddings toward target class (negative gradient)
        adv_embeddings = adv_embeddings - alpha * grad.sign()
        adv_embeddings = torch.max(torch.min(adv_embeddings, orig_embeddings + epsilon), orig_embeddings - epsilon)
        adv_embeddings = adv_embeddings.detach().requires_grad_(True)
    return adv_embeddings.detach()

def token_level_gcg_attack(model, input_ids, attention_mask, tokenizer, target_label, num_iter=10, method="mask"):
    """
    At each step, find the token position with the highest embedding gradient and replace it with [MASK] or a synonym.
    Returns: modified input_ids, list of modified positions
    """
    adv_input_ids = input_ids.clone().detach()
    modified_positions = []
    for _ in range(num_iter):
        # Get embeddings for current tokens
        embeddings = model.bert.embeddings(adv_input_ids).detach().clone().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label], device=logits.device))
        model.zero_grad()
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        loss.backward()
        grad = embeddings.grad.data.abs().sum(dim=2)[0]  # [seq_len], sum over embedding dims

        # Ignore special tokens ([CLS], [SEP], padding)
        mask = (adv_input_ids != tokenizer.cls_token_id) & \
               (adv_input_ids != tokenizer.sep_token_id) & \
               (adv_input_ids != tokenizer.pad_token_id)
        grad = grad * mask
        idx = torch.argmax(grad).item()
        if idx in modified_positions:
            break  # Stop if we've already modified this position

        orig_token_id = adv_input_ids[0, idx].item()
        if method == "mask":
            adv_input_ids[0, idx] = tokenizer.mask_token_id
        elif method == "synonym":
            orig_token = tokenizer.convert_ids_to_tokens([orig_token_id])[0]
            synonym = get_synonym(orig_token)
            adv_input_ids[0, idx] = tokenizer.convert_tokens_to_ids(synonym)
        modified_positions.append(idx)
        adv_input_ids = adv_input_ids.detach()
    return adv_input_ids, modified_positions

def true_token_level_gcg_attack(
    model, input_ids, attention_mask, tokenizer, target_label, num_iter=5, synonym_dict=None
):
    """
    At each step:
      - Find the token position with the highest embedding gradient.
      - Try replacing it with [MASK] and all synonyms (if provided).
      - Choose the replacement that most reduces the loss for the target class (targeted attack).
    Returns: modified input_ids, list of (position, old_token, new_token)
    """
    adv_input_ids = input_ids.clone().detach()
    modified_tokens = []

    for _ in range(num_iter):
        # 1. Get embeddings and compute gradient
        embeddings = model.bert.embeddings(adv_input_ids).detach().clone().requires_grad_(True)
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label], device=logits.device))
        model.zero_grad()
        if embeddings.grad is not None:
            embeddings.grad.zero_()
        loss.backward()
        grad = embeddings.grad.data.abs().sum(dim=2)[0]  # [seq_len]

        # 2. Ignore special tokens
        mask = (adv_input_ids != tokenizer.cls_token_id) & \
               (adv_input_ids != tokenizer.sep_token_id) & \
               (adv_input_ids != tokenizer.pad_token_id)
        grad = grad * mask
        idx = torch.argmax(grad).item()

        orig_token_id = adv_input_ids[0, idx].item()
        orig_token = tokenizer.convert_ids_to_tokens([orig_token_id])[0]

        # 3. Build candidate replacements
        candidates = []
        if synonym_dict and orig_token in synonym_dict:
            candidates.extend(synonym_dict[orig_token])
        candidates.append("[MASK]")  # Always try [MASK]
        candidates = list(set(candidates))  # Remove duplicates

        best_loss = float('inf')
        best_token = orig_token
        for cand in candidates:
            cand_id = tokenizer.convert_tokens_to_ids(cand)
            test_input_ids = adv_input_ids.clone()
            test_input_ids[0, idx] = cand_id
            with torch.no_grad():
                out = model(test_input_ids, attention_mask=attention_mask)
                logits = out.logits
                cand_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label], device=logits.device))
            if cand_loss < best_loss:
                best_loss = cand_loss
                best_token = cand

        # 4. If no improvement, stop
        if best_token == orig_token:
            break

        # 5. Apply best replacement
        adv_input_ids[0, idx] = tokenizer.convert_tokens_to_ids(best_token)
        modified_tokens.append((idx, orig_token, best_token))

    return adv_input_ids, modified_tokens

# Placeholder for REINFORCE-GCG
def reinforce_gcg_attack(*args, **kwargs):
    st.info("REINFORCE-GCG attack is a placeholder for future expansion.")

# --- UI: Sample texts and guidance ---

sample_texts = [
    "This movie was perfect.",
    "The plot was boring and predictable.",
    "I loved the acting and the story.",
    "The film was a waste of time.",
    "An outstanding and emotional experience.",
    "The direction was poor and the script was weak.",
]

st.markdown("**Try a sample review or enter your own:**")
col1, col2 = st.columns([2, 3])
with col1:
    sample = st.selectbox("Sample Reviews", [""] + sample_texts)
with col2:
    text = st.text_area("Your Review", value=sample if sample else "")

# --- UI: Attack parameters and target class ---

epsilon = st.slider("Epsilon (embedding attack strength)", min_value=0.01, max_value=0.2, value=0.1, step=0.01)
alpha = st.slider("Alpha (PGD step size)", min_value=0.001, max_value=0.02, value=0.01, step=0.001)
num_iter = st.slider("Number of PGD/GCG steps", min_value=1, max_value=30, value=10, step=1)
target_class = st.selectbox("Target class for attack", ["Positive", "Negative"])
target_label = get_target_label(target_class)

# --- Run attacks ---

if st.button("Run Attacks"):
    if not text.strip() or len(text.split()) < 3:
        st.warning("Please enter a longer review (at least 3 words).")
    else:
        try:
            # Tokenize and get embeddings
            encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            with torch.no_grad():
                embeddings = model.bert.embeddings(input_ids)

            # Original prediction
            with torch.no_grad():
                orig_outputs = model(input_ids, attention_mask=attention_mask)
                orig_pred = torch.softmax(orig_outputs.logits, dim=1)
                orig_class = torch.argmax(orig_pred).item()
                orig_conf = get_confidence(orig_pred, orig_class)

            # --- FGSM & PGD (embedding space) ---
            adv_embeddings_fgsm = fgsm_attack(model, embeddings, attention_mask, target_label, epsilon)
            adv_embeddings_pgd = pgd_attack(model, embeddings, attention_mask, target_label, epsilon, alpha, num_iter)

            def is_valid(tensor):
                return torch.isfinite(tensor).all().item()

            valid_fgsm = is_valid(adv_embeddings_fgsm)
            valid_pgd = is_valid(adv_embeddings_pgd)

            with torch.no_grad():
                adv_outputs_fgsm = model(inputs_embeds=adv_embeddings_fgsm, attention_mask=attention_mask) if valid_fgsm else None
                adv_outputs_pgd = model(inputs_embeds=adv_embeddings_pgd, attention_mask=attention_mask) if valid_pgd else None

            adv_pred_fgsm = torch.softmax(adv_outputs_fgsm.logits, dim=1) if valid_fgsm else None
            adv_pred_pgd = torch.softmax(adv_outputs_pgd.logits, dim=1) if valid_pgd else None

            adv_class_fgsm = torch.argmax(adv_pred_fgsm).item() if valid_fgsm else None
            adv_class_pgd = torch.argmax(adv_pred_pgd).item() if valid_pgd else None

            adv_conf_fgsm = get_confidence(adv_pred_fgsm, adv_class_fgsm) if valid_fgsm else None
            adv_conf_pgd = get_confidence(adv_pred_pgd, adv_class_pgd) if valid_pgd else None

            # --- Token-level GCG ---
            adv_input_ids_gcg, modified_positions = token_level_gcg_attack(
                model, input_ids, attention_mask, tokenizer, target_label, num_iter=num_iter, method="mask"
            )
            with torch.no_grad():
                adv_outputs_gcg = model(adv_input_ids_gcg, attention_mask=attention_mask)
                adv_pred_gcg = torch.softmax(adv_outputs_gcg.logits, dim=1)
                adv_class_gcg = torch.argmax(adv_pred_gcg).item()
                adv_conf_gcg = get_confidence(adv_pred_gcg, adv_class_gcg)

            # --- Detection: Confidence drop ---
            conf_drop_fgsm = orig_conf - adv_conf_fgsm if adv_conf_fgsm is not None else None
            conf_drop_pgd = orig_conf - adv_conf_pgd if adv_conf_pgd is not None else None
            conf_drop_gcg = orig_conf - adv_conf_gcg

            # --- Guidance and warnings ---
            if orig_conf > 0.95:
                st.warning("The model is very confident in its original prediction. "
                           "Try increasing epsilon or use a more ambiguous input for a more effective attack.")

            # --- Summary Table ---
            summary = []
            summary.append({
                "Attack": "Original",
                "Prediction": get_pred_label(orig_class),
                "Confidence": f"{orig_conf:.2f}",
                "Changed?": "—",
                "Conf Drop": "—"
            })
            summary.append({
                "Attack": "FGSM",
                "Prediction": get_pred_label(adv_class_fgsm) if valid_fgsm else "Invalid",
                "Confidence": f"{adv_conf_fgsm:.2f}" if valid_fgsm else "—",
                "Changed?": "✅" if valid_fgsm and adv_class_fgsm != orig_class else "❌",
                "Conf Drop": f"{conf_drop_fgsm:.2f}" if conf_drop_fgsm is not None else "—"
            })
            summary.append({
                "Attack": "PGD",
                "Prediction": get_pred_label(adv_class_pgd) if valid_pgd else "Invalid",
                "Confidence": f"{adv_conf_pgd:.2f}" if valid_pgd else "—",
                "Changed?": "✅" if valid_pgd and adv_class_pgd != orig_class else "❌",
                "Conf Drop": f"{conf_drop_pgd:.2f}" if conf_drop_pgd is not None else "—"
            })
            summary.append({
                "Attack": "Token-GCG",
                "Prediction": get_pred_label(adv_class_gcg),
                "Confidence": f"{adv_conf_gcg:.2f}",
                "Changed?": "✅" if adv_class_gcg != orig_class else "❌",
                "Conf Drop": f"{conf_drop_gcg:.2f}"
            })
            st.subheader("Prediction Summary")
            st.dataframe(pd.DataFrame(summary))

            # --- Parameter guidance ---
            for attack, changed, conf_drop in [
                ("FGSM", adv_class_fgsm != orig_class if valid_fgsm else False, conf_drop_fgsm),
                ("PGD", adv_class_pgd != orig_class if valid_pgd else False, conf_drop_pgd),
                ("Token-GCG", adv_class_gcg != orig_class, conf_drop_gcg)
            ]:
                if not changed:
                    st.info(f"{attack} did not flip the prediction. "
                            "Try increasing epsilon, alpha, or the number of steps for a stronger attack.")

            # --- Enhanced detection ---
            threshold = 0.3
            for attack, conf_drop in [
                ("FGSM", conf_drop_fgsm),
                ("PGD", conf_drop_pgd),
                ("Token-GCG", conf_drop_gcg)
            ]:
                if conf_drop is not None and conf_drop > threshold:
                    st.warning(f"{attack}: Large confidence drop detected ({conf_drop:.2f}). "
                               "This input may be adversarial.")

            # --- Visualization: Token-level GCG ---
            st.subheader("Token-GCG: Modified Tokens")
            orig_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            mod_tokens = tokenizer.convert_ids_to_tokens(adv_input_ids_gcg[0])
            token_table = pd.DataFrame({
                "Original Token": orig_tokens,
                "Modified Token": mod_tokens,
                "Changed?": ["✅" if i in modified_positions and orig_tokens[i] != mod_tokens[i] else "" for i in range(len(orig_tokens))]
            })
            st.dataframe(token_table)
            st.markdown("**Modified tokens highlighted:**")
            st.markdown(highlight_tokens(orig_tokens, mod_tokens), unsafe_allow_html=True)

            # --- Embedding mean and perturbation heatmaps ---
            st.subheader("Embedding Mean & Perturbation Heatmaps")
            orig_emb = embeddings.detach().numpy()[0]
            fgsm_emb = adv_embeddings_fgsm.detach().numpy()[0] if valid_fgsm else np.zeros_like(orig_emb)
            pgd_emb = adv_embeddings_pgd.detach().numpy()[0] if valid_pgd else np.zeros_like(orig_emb)
            fgsm_delta = fgsm_emb - orig_emb
            pgd_delta = pgd_emb - orig_emb

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            axs[0].plot(orig_emb.mean(axis=0), label="Original")
            if valid_fgsm: axs[0].plot(fgsm_emb.mean(axis=0), label="FGSM")
            if valid_pgd: axs[0].plot(pgd_emb.mean(axis=0), label="PGD")
            axs[0].set_title("Embedding Comparison (Mean)")
            axs[0].legend()
            im1 = axs[1].imshow(fgsm_delta, cmap='coolwarm', aspect='auto')
            plt.colorbar(im1, ax=axs[1], label='FGSM Perturbation')
            axs[1].set_title("FGSM Perturbations")
            axs[1].set_xlabel("Embedding Dimension")
            axs[1].set_ylabel("Token Position")
            im2 = axs[2].imshow(pgd_delta, cmap='coolwarm', aspect='auto')
            plt.colorbar(im2, ax=axs[2], label='PGD Perturbation')
            axs[2].set_title("PGD Perturbations")
            axs[2].set_xlabel("Embedding Dimension")
            axs[2].set_ylabel("Token Position")
            st.pyplot(fig)

            # --- GCG: Show which tokens were modified ---
            st.info(f"Token-GCG modified token positions: {modified_positions}")

            # --- REINFORCE-GCG Placeholder ---
            st.subheader("REINFORCE-GCG (Coming Soon)")
            reinforce_gcg_attack()

        except Exception as e:
            st.error(f"An error occurred: {e}")
