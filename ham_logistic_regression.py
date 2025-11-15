import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# Veri setini yükleyin ve hazırlayın
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

# Eksik değerleri doldurmak için SimpleImputer kullanın
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Veriyi eğitim ve test olarak ayırın
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Logistic Regression modelini eğitin ve değerlendirin
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrikleri hesaplayın
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

results = f"Logistic Regression:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n"

# Sonuçları bir dosyaya kaydedin
with open('C:/Users/User/Desktop/ML_Proje/b/ham_logistic_regression_results.txt', 'w') as f:
    f.write(results)

# Karışıklık matrisi için grafik oluşturun
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/ham_logistic_regression_confusion_matrix.png')
plt.close()

# Tahmin grafiği oluşturun
plt.scatter(X_test[:, 0], y_test, color='blue', label='Gerçek Değerler')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Tahminler')
plt.xlabel('Özellik 1')
plt.ylabel('Sonuç')
plt.legend()
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/ham_logistic_regression_image.png')
plt.close()
