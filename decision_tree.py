import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Tahmin ve performans metrikleri
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

# Metrikleri kaydet
output_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(output_dir, 'tree_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Doğruluk: {accuracy_tree}\n')
    f.write(f'Duyarlılık: {recall_tree}\n')
    f.write(f'Özgüllük: {precision_tree}\n')
    f.write(f'F1 Skoru: {f1_tree}\n')

# Karışıklık Matrisini Görselleştir
sns.heatmap(conf_matrix_tree, annot=True, fmt='d')
plt.title('Karar Ağaçları - Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.savefig(os.path.join(output_dir, 'tree_conf_matrix.png'))
plt.close()

# Performans Metriklerini Görselleştir
metrics = {'Accuracy': accuracy_tree, 'Precision': precision_tree, 'Recall': recall_tree, 'F1 Score': f1_tree}
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Karar Ağaçları - Performans Metrikleri')
plt.ylabel('Değer')
plt.savefig(os.path.join(output_dir, 'tree_performance_metrics.png'))
plt.close()

# Modeli kaydet
joblib.dump(decision_tree, os.path.join(output_dir, 'decision_tree_model.pkl'))
