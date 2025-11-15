import pandas as pd

# Ham veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# İlk birkaç satırı görüntüleyin
print(df.head())

# Temel bilgileri inceleyin
print(df.info())

# Eksik verileri kontrol edin
print(df.isnull().sum())
# Eksik verileri sıfır ile doldurun
df.fillna(0, inplace=True)

# Gereksiz sütunları kaldırın
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Kategorik değişkenleri sayısallaştırın
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# İşlenmiş veriyi kontrol edin
print(df.head())
print(df.info())
# Bağımlı ve bağımsız değişkenleri belirleyin
X = df.drop(columns=['Survived'])
y = df['Survived']

# Veriyi eğitim ve test setlerine ayırın
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test setlerinin boyutlarını kontrol edin
print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Lojistik Regresyon modeli
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Modelin tahminlerini kontrol edin
y_pred_logreg = logreg.predict(X_test)

# Performans metrikleri
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print(f'Lojistik Regresyon Doğruluk: {accuracy_logreg}')
print(f'Karışıklık Matrisi:\n {conf_matrix_logreg}')
print(f'Duyarlılık: {recall_logreg}')
print(f'Özgüllük: {precision_logreg}')
print(f'F1 Skoru: {f1_logreg}')
