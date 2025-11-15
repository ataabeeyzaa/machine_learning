import sys
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Modeli ve scaler'ı yükle
model = joblib.load('C:/Users/User/Desktop/ML_Proje/b/ros_random_forest_model.pkl')
scaler = joblib.load('C:/Users/User/Desktop/ML_Proje/b/scaler.pkl')
feature_names = joblib.load('C:/Users/User/Desktop/ML_Proje/b/feature_names.pkl')

# Komut satırından gelen parametreleri al
sinif = sys.argv[1]
yas = sys.argv[2]
kardes = sys.argv[3]
ebeveyn = sys.argv[4]
bilet = sys.argv[5]
cinsiyet = sys.argv[6]
binis = sys.argv[7]

# Giriş verilerini yazdır
print(f"Giriş Verileri - Sinif: {sinif}, Yas: {yas}, Kardes: {kardes}, Ebeveyn: {ebeveyn}, Bilet: {bilet}, Cinsiyet: {cinsiyet}, Binis: {binis}")

# Kullanıcı verisini oluştur
user_input = pd.DataFrame([[int(sinif), float(yas), int(kardes), int(ebeveyn), float(bilet), 
                            1 if cinsiyet == 'male' else 0, 1 if binis == 'S' else 0, 
                            1 if binis == 'C' else 0, 1 if binis == 'Q' else 0]], 
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_S', 'Embarked_C', 'Embarked_Q'])

# Kullanıcı girişini eğitim sırasında kullanılan öznitelik sıralamasına göre düzenleyin
user_input = user_input.reindex(columns=feature_names)

# Giriş verilerini tekrar yazdır
print(f"Güncellenmiş Giriş Verileri: \n{user_input}")

# Veriyi ölçekle
user_input_scaled = scaler.transform(user_input)

# Tahmin yap
proba = model.predict_proba(user_input_scaled)

# Sonuçları yazdır
print(f"Hayatta Kalma Olasılığı: {proba[0][1]:.4f}")
print(f"Hayatta Kalmama Olasılığı: {proba[0][0]:.4f}")
