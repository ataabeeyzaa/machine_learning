import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Komut satırından parametreleri al
sinif = sys.argv[1]
yas = sys.argv[2]
kardes = sys.argv[3]
ebeveyn = sys.argv[4]
bilet = sys.argv[5]
cinsiyet = sys.argv[6]
binis = sys.argv[7]

# Titanic veri setini yükleyelim
file_path = "C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv"
data = pd.read_csv(file_path)

# Eksik verileri dolduralım
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Özellikler (X) ve hedef değişken (y) belirleyelim
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
y = data['Survived']

# 'Sex' sütununu sayısal verilere dönüştürelim
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 'Embarked' sütununu dummies (one-hot encoding) ile dönüştürelim
X = pd.get_dummies(X, columns=['Embarked'], drop_first=False)

# Veriyi eğitim ve test kümelerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştıralım
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model oluşturma ve eğitme (class_weight='balanced' eklendi)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Kullanıcı verisini işleyelim
user_input = pd.DataFrame([[int(sinif), float(yas), int(kardes), int(ebeveyn), float(bilet), 
                            1 if cinsiyet == 'female' else 0, 1 if binis == 'S' else 0, 
                            1 if binis == 'C' else 0, 1 if binis == 'Q' else 0]], 
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_S', 'Embarked_C', 'Embarked_Q'])

# Kullanıcı verisi ile eğitim verisi sütunlarının uyumunu sağlayalım
user_input = user_input.reindex(columns=X.columns, fill_value=0)

# Kullanıcı verisini standartlaştıralım
user_input_scaled = scaler.transform(user_input)

# Tahmin yapalım
predicted_value = model.predict(user_input_scaled)
probabilities = model.predict_proba(user_input_scaled)

# Hayatta kalma ve ölüm olasılıkları
survival_prob = probabilities[0][1]
death_prob = probabilities[0][0]

# Çıktıyı formatla
output = f"""
Giriş Verileri:
Sinif: {sinif}, Yas: {yas}, Kardes: {kardes}, Ebeveyn: {ebeveyn}, 
Bilet: {bilet}, Cinsiyet: {cinsiyet}, Binis: {binis}

Tahmin Sonuçları:
Hayatta Kalma Olasılığı: {survival_prob:.4f}
Hayatta Kalmama Olasılığı: {death_prob:.4f}
Tahmin Edilen Sonuç: {'Hayatta Kalacak' if predicted_value[0] == 1 else 'Hayatta Kalmayacak'}
"""

# Çıktıyı yazdır
print(output)