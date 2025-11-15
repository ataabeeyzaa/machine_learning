import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Veri ve model yükleme
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

# Eğitim ve test setleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri ölçekleyin
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Model eğitimi
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

# Tahmin ve performans metrikleri
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

# Metrikleri kaydet
output_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(output_dir, 'logreg_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Doğruluk: {accuracy_logreg}\n')
    f.write(f'Duyarlılık: {recall_logreg}\n')
    f.write(f'Özgüllük: {precision_logreg}\n')
    f.write(f'F1 Skoru: {f1_logreg}\n')

# Karışıklık Matrisini Görselleştir
sns.heatmap(conf_matrix_logreg, annot=True, fmt='d')
plt.title('Lojistik Regresyon - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(os.path.join(output_dir, 'logreg_conf_matrix.png'))
plt.close()

# Performans Metriklerini Görselleştir
metrics = {'Accuracy': accuracy_logreg, 'Precision': precision_logreg, 'Recall': recall_logreg, 'F1 Score': f1_logreg}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Lojistik Regresyon - Performans Metrikleri')
plt.ylabel('Değer')
plt.savefig(os.path.join(output_dir, 'logreg_performance_metrics.png'))
plt.close()

# Modeli kaydet
joblib.dump(logreg, os.path.join(output_dir, 'logreg_model.pkl'))
