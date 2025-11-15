import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Titanic veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# İlk 5 satırı göster
print("İlk 5 Satır:")
print(df.head())

# Veri hakkında temel bilgiler (boş değerler, veri türleri vb.)
print("\nVeri Bilgileri:")
print(df.info())

# Sayısal değişkenlerdeki gürültü ve aykırı değerleri görselleştirmek için boxplot ve histogramlar
# Fare değişkeni için Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Fare'], color='skyblue')
plt.title("Fare Değişkeninin Box Plot'u")
plt.xlabel('Fare')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/fare_boxplot.png')
plt.close()

# Age değişkeni için Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title("Age Değişkeninin Box Plot'u")
plt.xlabel('Age')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/age_boxplot.png')
plt.close()

# Fare ve Age için Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['Fare'], kde=True, color='purple')
plt.title("Fare Değişkeninin Histogramı")
plt.xlabel('Fare')
plt.ylabel('Frekans')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/fare_histogram.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True, color='orange')
plt.title("Age Değişkeninin Histogramı")
plt.xlabel('Age')
plt.ylabel('Frekans')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/age_histogram.png')
plt.close()

# Sayısal kolonlarda aykırı değerleri temizlemek için Z-Score hesaplama
df_clean = df[['Age', 'Fare']].dropna()  # NaN olan satırları çıkartıyoruz
z_scores = stats.zscore(df_clean)

# Aykırı değerlerin (z-score > 3) temizlenmesi
df_cleaned = df_clean[(abs(z_scores) < 3).all(axis=1)]

# Temizlenmiş veri için scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_cleaned['Age'], y=df_cleaned['Fare'], color='green')
plt.title("Temizlenmiş Age ve Fare Dağılımı")
plt.xlabel('Age')
plt.ylabel('Fare')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/cleaned_scatter.png')
plt.close()

# Temizlenmiş Fare için Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['Fare'], color='lightgreen')
plt.title("Temizlenmiş Fare Değişkeninin Box Plot'u")
plt.xlabel('Fare')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/cleaned_fare_boxplot.png')
plt.close()

# Temizlenmiş Age için Box Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=df_cleaned['Age'], color='lightcoral')
plt.title("Temizlenmiş Age Değişkeninin Box Plot'u")
plt.xlabel('Age')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/cleaned_age_boxplot.png')
plt.close()

# Temizlenmiş Fare için Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Fare'], kde=True, color='orange')
plt.title("Temizlenmiş Fare Değişkeninin Histogramı")
plt.xlabel('Fare')
plt.ylabel('Frekans')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/cleaned_fare_histogram.png')
plt.close()

# Temizlenmiş Age için Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Age'], kde=True, color='blue')
plt.title("Temizlenmiş Age Değişkeninin Histogramı")
plt.xlabel('Age')
plt.ylabel('Frekans')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/cleaned_age_histogram.png')
plt.close()

# Kategorik veri (Embarked) üzerinde eksik değerleri doldurma (önceki adımda)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Kategorik verinin frekans dağılımı
plt.figure(figsize=(10, 6))
sns.countplot(x=df['Embarked'], palette='Set2')
plt.title("Embarked Değişkeninin Dağılımı")
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/embarked_countplot.png')
plt.close()

# Temizlenmiş veri kümesinin bilgilerini yazdırma
print("\nTemizlenmiş Veri Kümesi Bilgileri:")
print(df_cleaned.info())
