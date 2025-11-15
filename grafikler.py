import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ham veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Eksik verileri sıfır ile doldurun
df.fillna(0, inplace=True)

# Gereksiz sütunları kaldırın
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Kategorik değişkenleri sayısallaştırın
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Bağımlı ve bağımsız değişkenleri belirleyin
X = df.drop(columns=['Survived'])
y = df['Survived']

# Veriyi eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modeli
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Modelin tahminlerini kontrol edin
y_pred_rf = random_forest.predict(X_test)

# Performans metrikleri
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Karışıklık Matrisi Grafiği
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Hayatta Kalmadı', 'Hayatta Kaldı'], yticklabels=['Hayatta Kalmadı', 'Hayatta Kaldı'])
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.show()

# Diğer Performans Metrikleri Grafiği
metrics = {
    'Doğruluk': accuracy_rf,
    'Duyarlılık (Recall)': recall_rf,
    'Özgüllük (Precision)': precision_rf,
    'F1 Skoru': f1_rf
}

# Performans metriklerini bar grafiği ile görselleştirme
plt.figure(figsize=(8, 6))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.title('Model Performans Metrikleri')
plt.ylabel('Skor')
plt.show()

# Sonuçları yazdırma
print(f'Random Forest Doğruluk: {accuracy_rf}')
print(f'Karışıklık Matrisi:\n {conf_matrix_rf}')
print(f'Duyarlılık: {recall_rf}')
print(f'Özgüllük: {precision_rf}')
print(f'F1 Skoru: {f1_rf}')
