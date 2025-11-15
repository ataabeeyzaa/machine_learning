import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Titanic veri setini yükleyelim
file_path = r"C:\Users\User\Desktop\ML_Proje\archive\Titanic-Dataset.csv"
data = pd.read_csv(file_path)

# Eksik verileri forward fill yöntemiyle dolduralım
data.fillna(method='ffill', inplace=True)

# Özellikler (X) ve hedef değişkeni (y) belirleyelim
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']]
y = data['Survived']

# 'Sex' sütununu sayısal verilere dönüştürelim
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 'Embarked' sütununu dummies (one-hot encoding) ile dönüştürelim
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Veriyi eğitim ve test kümelerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştıralım
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Modelin doğruluğunu kontrol edelim
accuracy = model.score(X_train_scaled, y_train)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

# Cinsiyetin hayatta kalma üzerindeki etkisini görselleştirelim
sns.countplot(data=data, x='Sex', hue='Survived')
plt.title('Cinsiyete Göre Hayatta Kalma Oranı')
plt.show()

# Cinsiyet ve yaşın hayatta kalma üzerindeki etkisini görselleştirelim
sns.boxplot(data=data, x='Sex', y='Age', hue='Survived')
plt.title('Cinsiyet ve Yaşın Hayatta Kalma Üzerindeki Etkisi')
plt.show()

# Modelin eğitildiği ağırlıkları inceleyelim
print("Model Ağırlıkları (Coef):", model.coef_)

# Eğitim verisinin cinsiyet ve hayatta kalma üzerindeki etkisini kontrol edelim
gender_survival_rate = data.groupby('Sex')['Survived'].mean()
print("Cinsiyet ve Hayatta Kalma Oranı:")
print(gender_survival_rate)
