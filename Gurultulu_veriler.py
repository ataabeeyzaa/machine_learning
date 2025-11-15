# Gerekli kütüphaneleri yükleyin
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Titanic veri setini yükleyin (dosya yolunu belirtiyoruz)
file_path = r"C:\Users\User\Desktop\ML_Proje\archive\Titanic-Dataset.csv"
df = pd.read_csv(file_path)

# İlk 5 satırı görüntüleyin
print(df.head())

# Eksik değerlerin tespiti
print(df.isnull().sum())

# Eksik değerleri doldurmak için SimpleImputer kullanabiliriz
imputer = SimpleImputer(strategy='most_frequent')  # En sık görülen değeri kullanarak doldurur
df['Age'] = imputer.fit_transform(df[['Age']])

# Cinsiyet sütununu sayısal hale getirelim (Male = 0, Female = 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Kategorik değişkenler için One-Hot Encoding (Embarked, Cabin ve diğer kategorik değişkenler)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Verilerin gereksiz sütunlarını kaldırma (Örneğin, 'Name', 'Ticket', 'Cabin')
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Veriyi eğitim ve test setlerine ayıralım
X = df.drop('Survived', axis=1)  # Özellikler
y = df['Survived']  # Hedef değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özelliklerin standartlaştırılması
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modeli eğitelim (RandomForestClassifier örneği)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Modelin doğruluk skorunu değerlendirelim
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Sonuçları yazdıralım
print(f"Modelin doğruluk skoru: {accuracy * 100:.2f}%")

# Sonuçların metriklerini (confusion matrix vs.) görselleştirebiliriz
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Confusion matrix'i görselleştirelim
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Ayrıca ROC AUC gibi diğer metrikleri de hesaplayabiliriz
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC Skoru: {roc_auc:.2f}")
