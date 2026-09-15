from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

MODEL_PATH = Path(__file__).with_name("fraud.pkl")


def build_default_model():
    """Build a simple Bayesian model when the saved artifact is missing."""
    training_data = pd.DataFrame(
        {
            "type_code": [0, 0, 1, 1, 2, 2, 3, 4, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            "amt_log_disc": [0, 1, 2, 3, 4, 2, 3, 4, 1, 0, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 0, 2, 3, 1, 2, 3, 4, 0],
            "diff_orig_disc": [0, 1, 2, 3, 4, 1, 2, 3, 4, 0, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 0, 2, 3, 1, 2, 3, 4, 0],
            "diff_dest_disc": [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            "hour": [12, 20, 8, 17, 2, 5, 23, 9, 13, 18, 7, 15, 20, 21, 10, 11, 22, 1, 6, 16, 14, 19, 4, 3, 12, 20, 8, 17, 2],
            "isFraud": [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        }
    )

    model = BayesianNetwork(
        [
            ("type_code", "diff_orig_disc"),
            ("type_code", "isFraud"),
            ("type_code", "amt_log_disc"),
            ("amt_log_disc", "hour"),
            ("diff_orig_disc", "isFraud"),
            ("diff_orig_disc", "amt_log_disc"),
            ("diff_dest_disc", "type_code"),
            ("diff_dest_disc", "amt_log_disc"),
            ("diff_dest_disc", "isFraud"),
            ("diff_dest_disc", "diff_orig_disc"),
            ("diff_dest_disc", "hour"),
            ("isFraud", "hour"),
        ]
    )

    model.fit(
        training_data,
        estimator=MaximumLikelihoodEstimator,
        state_names={
            "type_code": [0, 1, 2, 3, 4],
            "amt_log_disc": [0, 1, 2, 3, 4],
            "diff_orig_disc": [0, 1, 2, 3, 4],
            "diff_dest_disc": [0, 1, 2, 3],
            "hour": list(range(24)),
            "isFraud": [0, 1],
        },
    )
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return build_default_model()


app = Flask(__name__)
model = load_model()
infer = VariableElimination(model)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    missing_fields = [field for field in ["type_code", "diff_orig_disc", "diff_dest_disc", "hour"] if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    amount_field = "amt_log_disc" if "amt_log_disc" in data else "amt_bin"
    if amount_field not in data:
        return jsonify({"error": "Missing amount field"}), 400

    evidence = {
        "type_code": int(data["type_code"]),
        "amt_log_disc": int(data[amount_field]),
        "diff_orig_disc": int(data["diff_orig_disc"]),
        "diff_dest_disc": int(data["diff_dest_disc"]),
        "hour": int(data["hour"]),
    }

    result = infer.query(variables=["isFraud"], evidence=evidence, show_progress=False)
    prob_fraud = float(result.values[1])
    return jsonify({"probability": prob_fraud})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
