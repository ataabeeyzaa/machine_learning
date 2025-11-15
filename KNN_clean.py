import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Gürültü eklenmiş Titanic veri setini yükle
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\b\\Titanic-Dataset-noisy.csv'
df = pd.read_csv(file_path)

# 1. Veriyi İnceleme
print("Veri Seti Genel Bilgisi:\n")
print(df.info())
print("\nEksik Değer Sayısı:\n", df.isnull().sum())

# 2. Kategorik Verileri Dönüştürme
# 'Sex' ve 'Embarked' sütunlarını sayısallaştır
categorical_mappings = {
    'Sex': {'male': 0, 'female': 1},
    'Embarked': {'C': 0, 'Q': 1, 'S': 2}
}
for column, mapping in categorical_mappings.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Sayısal ve kategorik sütunları ayır
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

# 3. Eksik Değerleri Doldurma
# Sayısal sütunlar için medyan stratejisi
num_imputer = SimpleImputer(strategy='median')
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

# Kategorik sütunlar için en sık görülen değer (mode) stratejisi
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

# 4. Uç Değer Analizi ve İşleme
# IQR yöntemi ile uç değerleri kaldırma
def remove_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

numerical_columns.remove('Survived')  # Hedef sütunu çıkar
df = remove_outliers(df, numerical_columns)

# 5. Özellik Normalizasyonu
scaler = StandardScaler()
features = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
target = df['Survived']
features_scaled = scaler.fit_transform(features)

# 6. Eğitim ve Test Verilerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# 7. KNN Modeli Eğitimi
model = KNeighborsClassifier(n_neighbors=5)  # K değerini burada belirleyin
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Performans Metrikleri
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 9. Sonuçları Yazdır ve Görselleştir
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Karışıklık Matrisi Görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('KNN Karışıklık Matrisi')
plt.show()

# Metrikleri kaydet
metrics = f'KNN:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n'
with open('C:\\Users\\User\\Desktop\\ML_Proje\\b\\knn_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics)

import joblib

# Modeli ve scaler'ı kaydet
joblib.dump(model, 'C:\\Users\\User\\Desktop\\ML_Proje\\model_knn.pkl')
joblib.dump(scaler, 'C:\\Users\\User\\Desktop\\ML_Proje\\scaler.pkl')



# Karışıklık matrisi kaydet
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('KNN Karışıklık Matrisi')
plt.savefig('C:\\Users\\User\\Desktop\\ML_Proje\\b\\knn_conf_matrix.png')
plt.close()
