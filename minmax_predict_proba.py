import sys
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Komut satırından gelen verileri al
sinif = sys.argv[1]
yas = sys.argv[2]
kardes = sys.argv[3]
ebeveyn = sys.argv[4]
bilet = sys.argv[5]
cinsiyet = sys.argv[6]
binis = sys.argv[7]

# Feature names ve model dosyalarını yükle
feature_names = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_feature_names.pkl')
model = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_random_forest_model.pkl')
scaler = joblib.load('C:/Users/User/Desktop/ML_Proje/b/minmax_scaler.pkl')

# Giriş verilerinin doğru şekilde işlenmesi
user_input = pd.DataFrame([[int(sinif),
                            1 if cinsiyet == 'female' else 0,  # Sex column: 0 if female, 1 if male
                            float(yas),
                            int(kardes),
                            int(ebeveyn),
                            float(bilet),
                            1 if binis == 'C' else 0,  # Embarked_C
                            1 if binis == 'Q' else 0,  # Embarked_Q
                            1 if binis == 'S' else 0]],  # Embarked_S
                          columns=feature_names)

# Debug print statements
print(f"Feature Names: {feature_names}")
print(f"Original Values: {sinif}, {yas}, {kardes}, {ebeveyn}, {bilet}, {cinsiyet}, {binis}")
print(f"User Input DataFrame:\n{user_input}")

# Veri yapısına uygunluğu kontrol et
if len(user_input.columns) != len(feature_names):
    raise ValueError(f"Veri girişi ile feature_names sütun sayısı eşleşmiyor: {len(user_input.columns)} != {len(feature_names)}")

user_input_scaled = user_input.copy()
user_input_scaled[['Age', 'Fare']] = scaler.transform(user_input[['Age', 'Fare']])  # Hem 'Age' hem de 'Fare' üzerinde ölçeklendirme yapılacak

# Tahminin olasılıklarını al
prediction_proba = model.predict_proba(user_input_scaled)

# Hayatta kalma ve kalmama olasılıkları
survival_prob = prediction_proba[0][1]  # Hayatta kalma olasılığı
death_prob = prediction_proba[0][0]  # Hayatta kalmama olasılığı

# Çıktıyı formatla
output = f"Giriş Verileri - Sinif: {sinif}, Yas: {yas}, Kardes: {kardes}, Ebeveyn: {ebeveyn}, Bilet: {bilet}, Cinsiyet: {cinsiyet}, Binis: {binis}\n"
output += f"Güncellenmiş Giriş Verileri:\n{user_input}\n"
output += f"Hayatta Kalma Olasılığı: {survival_prob:.4f}\n"
output += f"Hayatta Kalmama Olasılığı: {death_prob:.4f}"

print(output)
