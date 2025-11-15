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
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=False)  # drop_first=False ile tüm dummy değişkenler eklenir

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

# Tahmin ve performans metrikleri
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Metrikleri kaydet
output_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(output_dir, 'rf_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Doğruluk: {accuracy_rf}\n')
    f.write(f'Duyarlılık: {recall_rf}\n')
    f.write(f'Özgüllük: {precision_rf}\n')
    f.write(f'F1 Skoru: {f1_rf}\n')

# Karışıklık Matrisini Görselleştir
sns.heatmap(conf_matrix_rf, annot=True, fmt='d')
plt.title('Random Forest - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(os.path.join(output_dir, 'rf_conf_matrix.png'))
plt.close()

# Performans Metriklerini Görselleştir
metrics = {'Accuracy': accuracy_rf, 'Precision': precision_rf, 'Recall': recall_rf, 'F1 Score': f1_rf}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Random Forest - Performans Metrikleri')
plt.ylabel('Değer')
plt.savefig(os.path.join(output_dir, 'rf_performance_metrics.png'))
plt.close()

# Modeli kaydet
joblib.dump(random_forest, os.path.join(output_dir, 'random_forest_model.pkl'))

print(f"Model başarıyla kaydedildi: {os.path.join(output_dir, 'random_forest_model.pkl')}")
