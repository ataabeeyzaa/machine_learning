import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Eksik verileri sıfır ile doldurun
df.fillna(0, inplace=True)

# Gereksiz sütunları kaldırın
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Kategorik değişkenleri sayısallaştırın
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Özellikler ve hedef değişken
X = df.drop(columns=['Survived'])
y = df['Survived']

# SMOTE uygulayın
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Yeni sınıf dağılımını kontrol edin
print(pd.Series(y_resampled).value_counts())

# Dengesiz ve dengeli veri setleriyle model eğitimi ve değerlendirme
# Dengesiz veri setiyle eğitim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Dengesiz Veri Seti - Performans:")
print(classification_report(y_test, y_pred))

# Dengeli veri setiyle eğitim
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model_resampled.predict(X_test_resampled)
print("Dengeli Veri Seti - Performans:")
print(classification_report(y_test_resampled, y_pred_resampled))

# Karışıklık Matrisini Görselleştir (Dengesiz)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Dengesiz Veri Seti - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/imbalance_conf_matrix_unbalanced.png')
plt.close()

# Performans Metriklerini Görselleştir (Dengesiz)
metrics_unbalanced = classification_report(y_test, y_pred, output_dict=True)
selected_metrics_unbalanced = {m: metrics_unbalanced[m]['f1-score'] for m in metrics_unbalanced if m in ['0', '1']}
sns.barplot(x=list(selected_metrics_unbalanced.keys()), y=list(selected_metrics_unbalanced.values()))
plt.title('Dengesiz Veri Seti - Performans Metrikleri')
plt.xlabel('Metrik')
plt.ylabel('Değer')
plt.ylim(0, 1)
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/imbalance_performance_metrics_unbalanced.png')
plt.close()

# Karışıklık Matrisini Görselleştir (Dengeli)
conf_matrix_resampled = confusion_matrix(y_test_resampled, y_pred_resampled)
sns.heatmap(conf_matrix_resampled, annot=True, fmt='d')
plt.title('Dengeli Veri Seti - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/imbalance_conf_matrix_balanced.png')
plt.close()

# Performans Metriklerini Görselleştir (Dengeli)
metrics_balanced = classification_report(y_test_resampled, y_pred_resampled, output_dict=True)
selected_metrics_balanced = {m: metrics_balanced[m]['f1-score'] for m in metrics_balanced if m in ['0', '1']}
sns.barplot(x=list(selected_metrics_balanced.keys()), y=list(selected_metrics_balanced.values()))
plt.title('Dengeli Veri Seti - Performans Metrikleri')
plt.xlabel('Metrik')
plt.ylabel('Değer')
plt.ylim(0, 1)
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/imbalance_performance_metrics_balanced.png')
plt.close()
