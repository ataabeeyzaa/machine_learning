import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Veri yükleme
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Eksik verileri sıfır ile doldurun
df.fillna(0, inplace=True)

# Gereksiz sütunları kaldırın (Name, Ticket, Cabin, PassengerId sütunları gibi metin veri içeren sütunlar)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Kategorik değişkenleri sayısallaştırın
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Tüm dummy sütunların oluştuğundan emin olun
for col in ['Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']:
    if col not in df.columns:
        df[col] = 0

# Bağımlı ve bağımsız değişkenleri belirleyin
X = df.drop(columns=['Survived'])

# Özellikleri ölçekleyin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scaler'ı kaydet
output_dir = os.path.dirname(os.path.realpath(__file__))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

print("Scaler başarıyla kaydedildi.")
