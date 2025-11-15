import sys
import numpy as np
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict(model, input_features):
    input_array = np.array([input_features])
    prediction = model.predict(input_array)[0]
    prediction_proba = model.predict_proba(input_array)[0]
    return prediction, prediction_proba

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            raise ValueError("Not enough arguments provided. Expecting input features and mode.")
        
        input_features = [float(x) for x in sys.argv[1].split(",")]
        mode = sys.argv[2]
        model_path = "C:\\Users\\User\\Desktop\\ML_Proje\\b\\random_forest_model.pkl"
        
        model = load_model(model_path)

        if len(input_features) == 7:
            age, sibsp, parch, fare, sex, embarked = input_features[1:]
            
            # Dummy değişkenler
            sex_female = 1 if sex == 1 else 0
            sex_male = 0 if sex == 1 else 1
            embarked_C = 1 if embarked == 0 else 0
            embarked_Q = 1 if embarked == 1 else 0
            embarked_S = 1 if embarked == 2 else 0

            input_features = [age, sibsp, parch, fare, sex_female, sex_male, embarked_C, embarked_Q, embarked_S]

        prediction, prediction_proba = predict(model, input_features)

        # Negatif olasılıkları sıfıra zorla
        prediction_probability = max(0, prediction_proba[int(prediction)])

        result = f"{int(prediction)},{prediction_probability:.4f}"
        print(result)
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
