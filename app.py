from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def load_csv(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError(f"The file at {filepath} is empty.")
        return df
    except Exception as e:
        print(f"Error loading CSV file at {filepath}: {e}")
        return None

# Load datasets
df1 = load_csv("C:/Users/jahna/OneDrive/Desktop/HealthLinkHorizon/dataset.csv")
df2 = load_csv("C:/Users/jahna/OneDrive/Desktop/HealthLinkHorizon/Symptom-severity.csv")
df3 = load_csv("C:/Users/jahna/OneDrive/Desktop/HealthLinkHorizon/symptom_Description.csv")
df4 = load_csv("C:/Users/jahna/OneDrive/Desktop/HealthLinkHorizon/symptom_precaution.csv")

# Check if all dataframes are loaded correctly
if df1 is None or df2 is None or df3 is None or df4 is None:
    raise ValueError("One or more CSV files could not be loaded. Please check the file paths and contents.")

columns = [i for i in df1.iloc[:, 1:].columns]
temp = pd.melt(df1.reset_index(), id_vars=['index'], value_vars=columns)
temp['add1'] = 1
all_diseases = pd.pivot_table(temp, values='add1', index='index', columns='value')
all_diseases.insert(0, 'label', df1['Disease'])
all_diseases = all_diseases.fillna(0)

# Encoding the 'label' column to have numeric values
label_encoder = LabelEncoder()
all_diseases['label'] = label_encoder.fit_transform(all_diseases['label'])

# Splitting the data into features and labels
X = all_diseases.drop('label', axis=1)
y = to_categorical(all_diseases['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural Network Model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Dictionaries for severity and precautions
severity_dict = df2.set_index('Symptom').to_dict()['weight']
precautions_dict = df4.set_index('Disease').T.to_dict('list')

def predict_disease_with_probabilities_info(symptom_names):
    symptoms_input = [0] * len(X.columns)

    for symptom in symptom_names:
        if symptom in X.columns:
            symptoms_input[X.columns.get_loc(symptom)] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not recognized.")

    symptoms_array = np.array([symptoms_input])
    predicted_probs = model.predict(symptoms_array)[0]

    significant_probs = [(label_encoder.inverse_transform([i])[0], float(prob)) for i, prob in enumerate(predicted_probs) if prob > 0]
    significant_probs_sorted = sorted(significant_probs, key=lambda x: x[1], reverse=True)

    if len(significant_probs_sorted) == 0:
        print("No significant disease predictions based on the given symptoms.")
        return []

    results = []
    for disease, probability in significant_probs_sorted:
        if probability < 0.01:
            continue

        severity = str(severity_dict.get(disease, "Not available"))
        precautions = precautions_dict.get(disease, ["Not available"])
        formatted_precautions = ', '.join([str(p) for p in precautions if p not in [np.nan, "nan", None]])

        results.append({
            "disease": disease,
            "probability": f"{probability:.2f}",
            "severity": severity,
            "precautions": formatted_precautions
        })

    return results

@app.route('/')
def signup():
    return render_template("signup.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/forgotpassword')
def forgotpassword():
    return render_template("forgotpassword.html")

@app.route('/patientDetails')
def patientDetails():
    return render_template('patientDetails.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    prediction_results = predict_disease_with_probabilities_info(symptoms)
    return jsonify(prediction_results)

if __name__ == '__main__':
    app.run(debug=True)
