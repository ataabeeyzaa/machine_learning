import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Titanic veri setini belirtilen yoldan yükleyin
data_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\b\\Titanic-Dataset-noisy.csv'
data = pd.read_csv(data_path)

# Veriyi inceleyelim
print(data.head())

# Eksik sayısal verileri ortalama ile dolduralım
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns  # Sayısal sütunları seç
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())  # Sayısal verileri ortalama ile doldur

# Kategorik sütunları dolduralım (örneğin 'Embarked')
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # 'Embarked' sütununu mod ile doldur

# 'Sex' sütununu sayısal hale getirelim
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  # Kategorik 'Sex' sütununu sayısal hale getir

# 'Embarked' sütununu sayısal hale getirelim (Label Encoding kullanarak)
le = LabelEncoder()
data['Embarked'] = le.fit_transform(data['Embarked'])

# Gereksiz sütunları kaldırıyoruz (Name, Ticket, Cabin)
data = data.drop(columns=['Name', 'Ticket', 'Cabin'])

# Özellikler ve hedef değişkeni ayıralım
X = data.drop('Survived', axis=1)  # Özellikler
y = data['Survived']  # Hedef değişken

# Veriyi eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştırma (özellikle KNN için önemlidir)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modeli ile eğitim yapalım
knn = KNeighborsClassifier(n_neighbors=5)  # KNN modelini oluşturuyoruz
knn.fit(X_train, y_train)

# Tahmin yapalım
y_pred = knn.predict(X_test)

# Model sonuçlarını değerlendirelim
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Karışıklık Matrisi'ni görselleştirelim
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
