import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

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

# Karar Ağacı modeli
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Modelin tahminlerini kontrol edin
y_pred_tree = decision_tree.predict(X_test)

# Performans metrikleri
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
recall_tree = recall_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree)

print(f'Karar Ağacı Doğruluk: {accuracy_tree}')
print(f'Karışıklık Matrisi:\n {conf_matrix_tree}')
print(f'Duyarlılık: {recall_tree}')
print(f'Özgüllük: {precision_tree}')
print(f'F1 Skoru: {f1_tree}')
