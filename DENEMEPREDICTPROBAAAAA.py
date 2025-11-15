import pandas as pd
import joblib

# Feature names'i yükle
feature_names = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_feature_names.pkl')

# Modeli ve scaler'ı yükle
model = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_random_forest_model.pkl')
scaler = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_scaler.pkl')

# Yeni kullanıcı verisini hazırlama
user_input = pd.DataFrame({
    'Pclass': [1],  # Class
    'Sex': [1],  # Female (0 = Male, 1 = Female)
    'Age': [25.0],  # Age
    'SibSp': [0],  # Number of siblings/spouses aboard
    'Parch': [0],  # Number of parents/children aboard
    'Fare': [71.2833],  # Fare
    'Embarked_C': [1],  # Embarked: C (0 for others)
    'Embarked_Q': [0],  # Embarked: Q (0 for others)
    'Embarked_S': [0]   # Embarked: S (0 for others)
})

# Veriyi yazdırarak kontrol edelim
print("Yeni Veriniz:")
print(user_input)

# Eğitimde kullanılan feature isimlerini kontrol et
print("\nEğitimde Kullanılan Feature İsimleri:")
print(feature_names)

# Sadece 'Age' kolonu üzerinde ölçeklendirme yapalım
user_input_scaled = user_input.copy()
user_input_scaled[['Age']] = scaler.transform(user_input[['Age']])

# Ölçeklendirilmiş veriyi yazdırma
print("\nÖlçeklendirilmiş Veri:")
print(user_input_scaled)

# Modelle tahmin yap
prediction = model.predict(user_input_scaled)
probability = model.predict_proba(user_input_scaled)

# Sonuçları göster
print(f"\nTahmin Sonucu: {'Hayatta Kaldı' if prediction[0] == 1 else 'Hayatta Kalmadı'}")
print(f"Hayatta kalma olasılığı: {probability[0][1]:.4f}")
print(f"Hayatta kalmama olasılığı: {probability[0][0]:.4f}")
