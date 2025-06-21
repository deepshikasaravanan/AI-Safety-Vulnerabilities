from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Simulate sensitive training data
SENSITIVE_DATA = {
    "user_id": 12345,
    "email": "secretuser@example.com",
    "api_key": "ABC123SECRET"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        arr = np.array(data)
        if arr.shape[0] != 64:
            raise ValueError("Input must have 64 features")
        pred = int(np.sum(arr) % 10)
        # Insecure: accidentally leak sensitive info in the response!
        if pred == 7:  # Simulate a bug that leaks info for a certain output
            return jsonify({
                'prediction': pred,
                'debug': SENSITIVE_DATA  # <-- Sensitive info leak!
            })
        return jsonify({'prediction': pred})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5003)