import pandas as pd
import random
from faker import Faker

fake = Faker()

diagnoses = [
    "Hypertension", "Diabetes", "Asthma", "Anxiety", "Coronary Artery Disease",
    "Thyroid Disorder", "Chronic Kidney Disease", "Depression", "Hyperlipidemia", "Migraine"
]
medications = [
    "Lisinopril", "Metformin", "Albuterol", "Sertraline", "Aspirin",
    "Levothyroxine", "Losartan", "Fluoxetine", "Atorvastatin", "Sumatriptan"
]
allergies = ["None", "Penicillin", "Sulfa", "Latex", "Peanuts", "Shellfish"]
doctors = ["Dr. Adams", "Dr. Lee", "Dr. Patel", "Dr. Smith", "Dr. Johnson"]

num_records = 10000  # Exactly 10,000 records

data = []
for i in range(1001, 1001 + num_records):
    name = fake.name()
    age = random.randint(18, 90)
    gender = random.choice(["M", "F"])
    diagnosis = random.choice(diagnoses)
    medication = random.choice(medications)
    allergy = random.choice(allergies)
    doctor = random.choice(doctors)
    visit_date = fake.date_between(start_date='-2y', end_date='today')
    notes = fake.sentence(nb_words=8)
    data.append([
        i, name, age, gender, diagnosis, medication, allergy, doctor, visit_date, notes
    ])

df = pd.DataFrame(data, columns=[
    "PatientID", "Name", "Age", "Gender", "Diagnosis", "Medication",
    "Allergies", "Doctor", "VisitDate", "Notes"
])

df.to_csv("prompt-injection/data/synthetic_healthcare_data.csv", index=False)
print("Synthetic healthcare data with 10,000 records generated!")

df = pd.read_csv('ai-safety-vulnerabilities-demo/prompt-injection/data/synthetic_healthcare_data.csv')