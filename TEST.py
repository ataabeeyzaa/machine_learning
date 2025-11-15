import pandas as pd
import joblib

# Modeli, scaler'ı ve özellik adlarını yükle
model = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_random_forest_model.pkl')
scaler = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_scaler.pkl')
feature_names = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_feature_names.pkl')

# Kullanıcıdan alınan veriler
sinif = 1
yas = 25
kardes = 2
ebeveyn = 1
bilet = 70
cinsiyet = 'male'
binis = 'S'

# Eğitim sırasında kullanılan özellik adlarını belirleyin
scaler_features = feature_names

# Kullanıcı girdisini DataFrame'e çevirin
user_input = pd.DataFrame([[int(sinif), float(yas), int(kardes), int(ebeveyn), float(bilet),
                            1 if cinsiyet == 'male' else 0,
                            1 if binis == 'S' else 0,  # Embarked_S
                            1 if binis == 'C' else 0,  # Embarked_C
                            1 if binis == 'Q' else 0]],  # Embarked_Q
                          columns=scaler_features)

# Giriş verisini yazdır
print("Giriş Verisi:")
print(user_input)  # Giriş verisini kontrol etmek için

# Eksik sütunları ekleyin (Gerekirse)
for col in scaler_features:
    if col not in user_input.columns:
        user_input[col] = 0  # ya da uygun bir varsayılan değer

# Giriş verisini tekrar yazdır
print("Güncellenmiş Giriş Verisi:")
print(user_input)  # Düzenlenmiş veriyi kontrol etmek için

# Kullanıcı girdisini ölçekle
user_input_scaled = scaler.transform(user_input[scaler_features])

# Ölçeklenmiş girdiyi yazdır
print("Ölçeklenmiş Giriş Verisi:")
print(user_input_scaled)

# Tahmin yap
prediction_proba = model.predict_proba(user_input_scaled)

# Tahmin sonuçlarını yazdır
print("\nTahmin Sonuçları:")
print(f"Hayatta kalma olasılığı: {prediction_proba[0][1]:.4f}")
print(f"Hayatta kalmama olasılığı: {prediction_proba[0][0]:.4f}")
