import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler():
    output_dir = os.path.dirname(os.path.realpath(__file__))
    model = joblib.load(os.path.join(output_dir, 'random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(output_dir, 'scaler.pkl'))
    return model, scaler

def main():
    try:
        # Komut satırı argümanlarını al
        input_data = sys.argv[1]
        mode = sys.argv[2]

        # Giriş verilerini işleme
        input_data = list(map(float, input_data.split(',')))
        input_df = pd.DataFrame([input_data], columns=[
            'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
        ])

        # Model ve scaler'ı yükle
        model, scaler = load_model_and_scaler()

        # Veriyi ölçeklendir
        input_df_scaled = scaler.transform(input_df)

        # Tahmin proba
        proba = model.predict_proba(input_df_scaled)
        prediction = model.predict(input_df_scaled)

        # Olasılık değerlerini kontrol et
        survival_probability = proba[0][1]
        if survival_probability < 0 or survival_probability > 1:
            raise ValueError(f'Invalid probability value: {survival_probability}')

        # Sonucu yazdır
        print(f'{prediction[0]},{survival_probability}')

    except Exception as e:
        print(f'Hata: {e}')

if __name__ == '__main__':
    main()
