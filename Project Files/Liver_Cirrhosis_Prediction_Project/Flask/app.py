from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and normalizer from root directory
model = pickle.load(open("rf_acc_68.pkl", "rb"))
normalizer = pickle.load(open("normalizer.pkl", "rb"))

# Exact 34 feature order
feature_order = [
    'Age', 'Gender', 'Duration of alcohol consumption(years)',
    'Quantity of alcohol consumption (quarters/day)', 'Hepatitis B infection',
    'Hepatitis C infection', 'Diabetes Result', 'Blood pressure (mmhg)',
    'Obesity', 'Family history of cirrhosis/ hereditary', 'TCH', 'TG', 'LDL',
    'HDL', 'Hemoglobin  (g/dl)', 'PCV  (%)', 'MCV   (femtoliters/cell)',
    'Total Count', 'Polymorphs  (%) ', 'Lymphocytes  (%)', 'Monocytes   (%)',
    'Eosinophils   (%)', 'Basophils  (%)', 'Platelet Count  (lakhs/mm)',
    'Total Bilirubin    (mg/dl)', 'Direct    (mg/dl)', 'Indirect     (mg/dl)',
    'Total Protein     (g/dl)', 'Albumin   (g/dl)', 'Globulin  (g/dl)',
    'A/G Ratio', 'AL.Phosphatase      (U/L)', 'SGOT/AST      (U/L)',
    'SGPT/ALT (U/L)'
]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Dropdown and string mappings
        binary_map = {
            'male': 1, 'female': 0,
            'yes': 1, 'no': 0,
            'positive': 1, 'negative': 0
        }

        input_data = {}
        for feature in feature_order:
            raw_value = request.form.get(feature, "").strip().lower()
            if raw_value in binary_map:
                input_data[feature] = binary_map[raw_value]
            elif "/" in raw_value:  # e.g., blood pressure "120/80"
                input_data[feature] = float(raw_value.split("/")[0])
            else:
                try:
                    input_data[feature] = float(raw_value)
                except:
                    input_data[feature] = 0  # fallback

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Normalize
        input_norm = normalizer.transform(input_df)

        # Predict
        prediction = model.predict(input_norm)[0]
        if prediction == 1:
            result = "⚠ The patient is at risk of liver cirrhosis. Please consult a doctor."
        else:
            result = "✅ No signs of liver cirrhosis based on the given data."

        return render_template("inner-page.html", result=result)

    except Exception as e:
        return render_template("inner-page.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)