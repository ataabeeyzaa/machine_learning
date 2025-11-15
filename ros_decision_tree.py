import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

# Veri setini yükleyin ve hazırlayın
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

# ROS uygulayın
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Dengesiz veri setiyle eğitim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics_unbalanced = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
}
conf_matrix_unbalanced = confusion_matrix(y_test, y_pred)

# Dengeli veri setiyle eğitim
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model_resampled = DecisionTreeClassifier(random_state=42)
model_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model_resampled.predict(X_test_resampled)
metrics_balanced = {
    'accuracy': accuracy_score(y_test_resampled, y_pred_resampled),
    'precision': precision_score(y_test_resampled, y_pred_resampled),
    'recall': recall_score(y_test_resampled, y_pred_resampled),
    'f1_score': f1_score(y_test_resampled, y_pred_resampled)
}
conf_matrix_balanced = confusion_matrix(y_test_resampled, y_pred_resampled)

# Metrikleri kaydet
output_dir = 'C:/Users/User/Desktop/ML_Proje/b'
with open(os.path.join(output_dir, 'ros_metrics_unbalanced_decision_tree.txt'), 'w') as f:
    for metric, value in metrics_unbalanced.items():
        f.write(f'{metric}: {value}\n')
with open(os.path.join(output_dir, 'ros_metrics_balanced_decision_tree.txt'), 'w') as f:
    for metric, value in metrics_balanced.items():
        f.write(f'{metric}: {value}\n')

# Karışıklık Matrisi Görselleştir (Dengesiz)
sns.heatmap(conf_matrix_unbalanced, annot=True, fmt='d')
plt.title('Dengesiz Veri Seti - Karışıklık Matrisi (ROS - Karar Ağacı)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(os.path.join(output_dir, 'ros_conf_matrix_unbalanced_decision_tree.png'))
plt.close()

# Performans Metrikleri Görselleştir (Dengesiz)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(metrics_unbalanced.keys()), y=list(metrics_unbalanced.values()))
plt.title('Dengesiz Veri Seti - Performans Metrikleri (ROS - Karar Ağacı)')
plt.xlabel('Metrik')
plt.ylabel('Değer')
plt.ylim(0, 1)
plt.savefig(os.path.join(output_dir, 'ros_metrics_unbalanced_decision_tree.png'))
plt.close()

# Karışıklık Matrisi Görselleştir (Dengeli)
sns.heatmap(conf_matrix_balanced, annot=True, fmt='d')
plt.title('Dengeli Veri Seti - Karışıklık Matrisi (ROS - Karar Ağacı)')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(os.path.join(output_dir, 'ros_conf_matrix_balanced_decision_tree.png'))
plt.close()

# Performans Metrikleri Görselleştir (Dengeli)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(metrics_balanced.keys()), y=list(metrics_balanced.values()))
plt.title('Dengeli Veri Seti - Performans Metrikleri (ROS - Karar Ağacı)')
plt.xlabel('Metrik')
plt.ylabel('Değer')
plt.ylim(0, 1)
plt.savefig(os.path.join(output_dir, 'ros_metrics_balanced_decision_tree.png'))
plt.close()
