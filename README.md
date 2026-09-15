# Online Payment Fraud Prediction

## Problem Statement

Online payment fraud is a significant issue in digital banking and e-commerce systems, where fraudulent transactions can lead to financial losses, customer distrust, and reputational damage. The challenge is to detect suspicious transactions early and estimate the likelihood that a payment is fraudulent based on transaction characteristics.

This project addresses that problem by developing a probabilistic fraud detection model using a Bayesian Network. Instead of using a traditional black-box classifier, the model learns relationships among transaction features and estimates the probability of fraud under uncertainty.

## Project Objective

The main objective of this project is to build a prediction system that can:

- analyze transaction patterns
- identify suspicious behaviors
- estimate the probability of fraud for a new transaction
- provide a simple user-friendly interface for prediction

## Why Bayesian Networks?

Bayesian Networks are useful for real-world decision problems because they model dependencies between variables and support probabilistic reasoning. In this project, features such as transaction type, amount range, deviation from normal behavior, and transaction hour are treated as evidence, and the model computes the probability that the transaction is fraudulent.

## Features Used in the Model

The system uses transaction-related variables such as:

- transaction type (`type_code`)
- amount category (`amt_log_disc` / `amt_bin`)
- difference from normal origin pattern (`diff_orig_disc`)
- difference from normal destination pattern (`diff_dest_disc`)
- hour of transaction (`hour`)
- fraud label (`isFraud`)

## Methodology

1. Collect transaction data containing both normal and fraudulent examples.
2. Preprocess and discretize relevant fields.
3. Learn a Bayesian Network structure from the data.
4. Fit conditional probability distributions using Maximum Likelihood Estimation.
5. Use probabilistic inference to predict fraud probability for new transactions.
6. Deploy the model through a Flask web application.

## Tech Stack

- Python
- Flask
- pgmpy
- pandas
- joblib
- HTML/CSS/JavaScript

## Project Structure

- `app.py` — Flask application and prediction logic
- `templates/index.html` — web interface for entering transaction details
- `bayesian.ipynb` — training and model-building notebook
- `requirements.txt` — project dependencies
- `fraud.pkl` — trained Bayesian model file (created automatically if missing)

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the application

```bash
python app.py
```

### 3. Open the app in browser

```text
http://127.0.0.1:5000
```

## Sample API Request

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type_code": 0,
    "amt_bin": 2,
    "diff_orig_disc": 3,
    "diff_dest_disc": 0,
    "hour": 20
  }'
```

## Sample Response

```json
{
  "probability": 0.010844260589595322
}
```

## Conclusion

This project demonstrates how probabilistic graphical models can be applied to real-world fraud detection problems. It combines statistical reasoning, machine learning, and web deployment into a practical system that estimates transaction fraud risk in a transparent and interpretable way.

## Note

The repository originally did not include a saved `fraud.pkl` file, so the application includes a fallback mechanism that builds a default Bayesian model if the trained file is missing. For better performance, the model should be retrained on a larger dataset using the notebook in `bayesian.ipynb`.

