from flask import Flask, jsonify, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

app = Flask(__name__)

# Model yükleme
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_model = pickle.load(file)

with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data.get('model')
    features = np.array(data.get('features')).reshape(1, -1)

    if model_name == "logistic":
        model = logistic_model
    elif model_name == "decision_tree":
        model = decision_tree_model
    elif model_name == "random_forest":
        model = random_forest_model
    else:
        return jsonify({'error': 'Invalid model name'})

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features).tolist()

    return jsonify({
        'prediction': int(prediction),
        'probabilities': probabilities
    })

@app.route('/metrics', methods=['POST'])
def metrics():
    data = request.json
    model_name = data.get('model')
    y_true = data.get('y_true')
    y_pred = data.get('y_pred')

    if model_name == "logistic":
        model = logistic_model
    elif model_name == "decision_tree":
        model = decision_tree_model
    elif model_name == "random_forest":
        model = random_forest_model
    else:
        return jsonify({'error': 'Invalid model name'})

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name.capitalize()} Confusion Matrix')
    plt.savefig(f'{model_name}_confusion_matrix.png')

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)
