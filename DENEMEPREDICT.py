import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Veri ve model yükleme
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Eksik verileri sıfır ile doldurun
df.fillna(0, inplace=True)

# Gereksiz sütunları kaldırın (Name, Ticket, Cabin sütunları gibi metin veri içeren sütunlar)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Kategorik değişkenleri sayısallaştırın
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Bağımlı ve bağımsız değişkenleri belirleyin
X = df.drop(columns=['Survived'])
y = df['Survived']

# Eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçekleyin
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model eğitimi
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Kullanıcıdan giriş verisi alalım
user_input = pd.DataFrame({
    'Pclass': [1],
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [50],
    'Sex_male': [1],  # Cinsiyet: Male -> 1
    'Embarked_S': [1],  # Embarked: S -> 1
    'Embarked_C': [0],  # Embarked: C -> 0 (eksik sütunları ekledik)
    'Embarked_Q': [0]   # Embarked: Q -> 0 (eksik sütunları ekledik)
})

# Eğitimde kullanılan kolon sırasını alalım
train_columns = X.columns

# Kullanıcı girişi sütunlarını doğru sıraya koyun
user_input = user_input[train_columns]

# Özellikleri ölçekleyin
user_input_scaled = scaler.transform(user_input)

# Modelle olasılık tahmini yapın
probabilities = random_forest.predict_proba(user_input_scaled)

# Sonuçları yazdır
print(f"Hayatta Kalma Olasılığı: {probabilities[0][1]:.4f}")
print(f"Hayatta Kalmama Olasılığı: {probabilities[0][0]:.4f}")
