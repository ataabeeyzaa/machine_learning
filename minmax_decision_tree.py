import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score

# Titanic veri setini yükle
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\archive\\Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Veriyi temizle (Eksik verileri doldur, kategorik verileri dönüştür)
df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Gereksiz kolonları kaldır
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Eksik Embarked verilerini doldur
df['Age'] = df['Age'].fillna(df['Age'].median())  # Eksik Age verilerini doldur
df['Fare'] = df['Fare'].fillna(df['Fare'].median())  # Eksik Fare verilerini doldur


scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])  # 'Age' ve 'Fare'


# Kategorik verileri dönüştür
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Kategorik veriyi sayısala dönüştür
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})  # Kategorik veriyi sayısala dönüştür

# Özellikler (X) ve hedef değişken (y) olarak ayır
X = df.drop('Survived', axis=1)  # 'Survived' dışında tüm kolonlar
y = df['Survived']  # 'Survived' hedef değişkeni

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi normalleştir (Fare gibi diğer sayısal kolonlara normalizasyon yapılmadı)
# Fare gibi diğer sayısal özelliklere normalizasyon yapılmadı.

# Karar ağacı modelini oluştur
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

# Metrikleri kaydet
metrics = f'decision_tree:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\nROC AUC Score: {roc_auc}'
with open('C:\\Users\\User\\Desktop\\ML_Proje\\b\\tree_metrics.txt', 'w', encoding='utf-8') as f:
    f.write(metrics)

# Karışıklık Matrisi Grafiği
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hayatta Kalmayan', 'Hayatta Kalan'], yticklabels=['Hayatta Kalmayan', 'Hayatta Kalan'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karar Ağacı Karışıklık Matrisi')
plt.savefig('C:\\Users\\User\\Desktop\\ML_Proje\\b\\tree_conf_matrix.png')
plt.close()
