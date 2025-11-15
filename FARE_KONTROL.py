import numpy as np
import pandas as pd
import joblib
import os

# Modeli yükleyin
output_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(output_dir, 'random_forest_model.pkl')
random_forest = joblib.load(model_path)

# Örnek veri oluşturun (modelin eğitim sırasında beklediği özellikleri kullanarak)
example_data = pd.DataFrame({
    'Pclass': [3, 1],  # Örnek sınıf (1: birinci sınıf, 3: üçüncü sınıf)
    'Age': [22, 38],   # Örnek yaş
    'SibSp': [1, 1],   # Kardeş/eş sayısı
    'Parch': [0, 0],   # Çocuk/ebeveyn sayısı
    'Fare': [7.25, 71.2833],  # Bilet ücreti
    'Sex_male': [1, 0],  # Erkek mi? (1: Erkek, 0: Kadın)
    'Embarked_Q': [0, 0],  # Embarked Q (1: Evet, 0: Hayır)
    'Embarked_S': [1, 0]   # Embarked S (1: Evet, 0: Hayır)
})

# Eğitim sırasında kullanılan scaler'ı yeniden kullanarak özellikleri ölçeklendirin
scaler_path = os.path.join(output_dir, 'scaler.pkl')
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
if scaler:
    example_data = scaler.transform(example_data)

# Olasılık tahmini yapın
proba_predictions = random_forest.predict_proba(example_data)

# Tahmin olasılıklarını görüntüleyin
for i, proba in enumerate(proba_predictions):
    print(f"Örnek {i+1}: Hayatta kalma olasılığı: {proba[1]:.2f}, Hayatta kalmama olasılığı: {proba[0]:.2f}")
