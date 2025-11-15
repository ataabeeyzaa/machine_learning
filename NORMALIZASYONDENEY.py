import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Titanic veri setini yükle
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\archive\\Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Veriyi temizle (Eksik verileri doldur, kategorik verileri dönüştür)
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Gereksiz kolonları kaldır
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Eksik Embarked verilerini doldur
df['Age'] = df['Age'].fillna(df['Age'].median())  # Eksik Age verilerini doldur
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Eksik Fare verilerini doldur
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Kategorik veriyi sayısala dönüştür
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Kategorik veriyi sayısala dönüştür

# Özellikler (X) ve hedef değişken (y) olarak ayır
X = df.drop('Survived', axis=1)  # 'Survived' dışında tüm kolonlar
y = df['Survived']  # 'Survived' hedef değişkeni

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fare normalizasyonu için Z-Score (StandardScaler) kullan
zscore_scaler = StandardScaler()
X_train['Fare'] = zscore_scaler.fit_transform(X_train[['Fare']])
X_test['Fare'] = zscore_scaler.transform(X_test[['Fare']])

# Karar Ağacı modelini oluştur ve eğit
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Test setinde tahmin yap
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Sonuçları terminalde göster
print("\nDecision Tree Metrics (Z-Score Normalization - Fare):")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
