from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder="template")
model = tf.keras.models.load_model("cardiac_arrest.h5")

# Label mapping (ensure this matches your model's classes)
label_mapping = {0: "Normal", 1: "Arrhythmia Type 1", 2: "Arrhythmia Type 2"}

# Default values for non-derived features (update as needed)
default_values = {
    "age": 40,
    "sex": 0,  # 0: Male, 1: Female
    "height": 170,  # cm
    "weight": 70,  # kg
    "qrs_duration": 90.0,
    "p-r_interval": 160.0,
    "q-t_interval": 400.0,
    "t_interval": 200.0,
    "p_interval": 100.0,
    "qrs": 0,
    "T": 0,
    "P": 0,
    "QRST": 0,
    "J": 0,
    "heart_rate": 75.0,
    "q_wave": 0,
    "r_wave": 0,
    "s_wave": 0,
    "R'_wave": 0,
    "S'_wave": 0,
}

# Correct feature order (22 features including 'bmi' and 'qtc')
training_features = [
    "age",
    "sex",
    "height",
    "weight",
    "qrs_duration",
    "p-r_interval",
    "q-t_interval",
    "t_interval",
    "p_interval",
    "qrs",
    "T",
    "P",
    "QRST",
    "J",
    "heart_rate",
    "q_wave",
    "r_wave",
    "s_wave",
    "R'_wave",
    "S'_wave",
    "bmi",
    "qtc",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = {}

    # Collect form inputs or use defaults
    for feature in training_features:
        if feature in ["bmi", "qtc"]:  # Skip derived features
            continue
        form_value = request.form.get(feature)
        if form_value not in (None, ""):
            try:
                features[feature] = float(form_value)
            except:
                features[feature] = default_values.get(feature, 0)
        else:
            features[feature] = default_values.get(feature, 0)

    # Calculate BMI
    features["bmi"] = features["weight"] / ((features["height"] / 100) ** 2)

    # Calculate QTc (correct formula from your model)
    features["qtc"] = features["q-t_interval"] / np.sqrt(
        features["p-r_interval"] / 1000
    )

    # Build feature vector in correct order
    feature_vector = [features[feat] for feat in training_features]

    # Predict
    input_data = np.array([feature_vector])
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    diagnosis = label_mapping.get(predicted_class, "Unknown")

    return render_template("index.html", prediction=diagnosis)


if __name__ == "__main__":
    app.run(debug=True)
