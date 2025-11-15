import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Titanic veri setini yükleme
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\archive\\Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Kategorik verileri dönüştürme
categorical_mappings = {
    'Sex': {'male': 0, 'female': 1},
    'Embarked': {'C': 0, 'Q': 1, 'S': 2}
}
for column, mapping in categorical_mappings.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Eksik değerleri doldurma (interpolasyon - mean)
df = df.interpolate(method='linear', limit_direction='forward', axis=0)

# Eksik değerler kalmadığını kontrol et
print("Eksik Değer Kontrolü:\n", df.isnull().sum())

# Özellik ve hedef değişkeni ayırma
features = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'], errors='ignore')
target = df['Survived']

# Özelliklerin normalizasyonu
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Eğitim ve test verilerine ayırma
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Lojistik regresyon modeli oluşturma ve eğitme
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans değerlendirme
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Sonuçları yazdırma
print("Karışıklık Matrisi:\n", conf_matrix)
print("\nDoğruluk Skoru:", accuracy)
print("\nSınıflandırma Raporu:\n", report)
