import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SequentialFeatureSelector

# Titanic veri setini yükleyelim
df = pd.read_csv("C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv")

# Önemli sütunları seçelim (örnek veri işleme)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # Kullanılabilir sütunlar
df = df.dropna()  # Eksik verileri çıkaralım
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Cinsiyet sütununu sayısallaştıralım

# Bağımlı ve bağımsız değişkenleri ayıralım
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]  # Özellikler
y = df['Survived']  # Hedef değişken

# Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik Regresyon Modeli Tanımla
model = LogisticRegression(max_iter=500)

# Sequential Feature Selector ile İleri Seçim
sfs = SequentialFeatureSelector(model, n_features_to_select=3, direction='forward', scoring='accuracy', cv=5)
sfs.fit(X_train, y_train)

# Seçilen özellikler
selected_features = X_train.columns[sfs.get_support()]
print("Seçilen Özellikler:", list(selected_features))

# Yeni özellik kümesiyle model eğitimi
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# Model Performansı
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu (Accuracy):", accuracy)
