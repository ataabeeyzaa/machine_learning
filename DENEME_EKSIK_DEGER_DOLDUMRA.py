import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükleme
file_path = r'C:\Users\User\Desktop\ML_Proje\archive\Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Gürültü ekleme fonksiyonu (sadece sayısal özelliklere)
def add_noise(df, noise_level=0.1):
    # Sayısal özellikleri seçme (Survived sütunu hariç)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = numerical_cols[numerical_cols != 'Survived']  # 'Survived' sütununu hariç tutuyoruz
    
    # Gürültü oranı kadar rastgele gürültü ekleyelim
    noise = np.random.normal(0, noise_level, size=df[numerical_cols].shape)
    
    # Veri setine gürültü ekleyelim
    df[numerical_cols] = df[numerical_cols] + noise
    
    return df

# Eksik verileri kontrol eden fonksiyon
def check_missing_values(df):
    print("\nMissing values count after processing:")
    print(df.isna().sum())

# Logistic Regression modelini oluşturma ve değerlendirme fonksiyonu
def evaluate_classification_model(df):
    # Eksik verileri kontrol etme
    check_missing_values(df)

    # 'Survived' sütununu hedef değişken olarak seçme
    X = df.drop('Survived', axis=1).select_dtypes(include=[np.number])  # Sadece sayısal kolonlar
    y = df['Survived']

    # Veriyi eğitim ve test setine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression modelini oluşturma ve eğitme
    model = LogisticRegression(max_iter=1000)
    
    # Modeli eğitmeden önce NaN değerleri kontrol etme
    if X_train.isna().sum().sum() > 0:
        print("NaN values detected in training set. Filling with median...")
        X_train = X_train.fillna(X_train.median())
    if X_test.isna().sum().sum() > 0:
        print("NaN values detected in test set. Filling with median...")
        X_test = X_test.fillna(X_test.median())
    
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)
    
    # Karışıklık matrisini görselleştirme
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

# Titanic veri setine gürültü ekleyelim (gürültü sadece özelliklere eklenmeli)
df_noisy = add_noise(df.copy(), noise_level=0.1)

# Eksik veri oluşturmak için bazı verileri kaybedelim (örneğin, rastgele %10'u eksik olsun)
missing_percentage = 0.1
for col in df_noisy.columns:
    if df_noisy[col].dtype == np.number:  # Sayısal sütunlarda işlem yapalım
        df_noisy.loc[df_noisy.sample(frac=missing_percentage).index, col] = np.nan

# A) Delete Rows with Missing Values
df_cleaned = df_noisy.dropna()
print("Evaluating with Deleted Missing Values (After Adding Noise)")
evaluate_classification_model(df_cleaned)

# B) Replacing with previous value - Forward fill
df_ffill = df_noisy.fillna(method="ffill")
print("Evaluating with Forward Fill (After Adding Noise)")
evaluate_classification_model(df_ffill)

# C) Replacing with next value - Backward fill
df_bfill = df_noisy.fillna(method="bfill")
print("Evaluating with Backward Fill (After Adding Noise)")
evaluate_classification_model(df_bfill)

# D) Filling missing values with 0
df_zero_fill = df_noisy.fillna(0)
print("Evaluating with Zero Fill (After Adding Noise)")
evaluate_classification_model(df_zero_fill)

# E) Interpolate missing values (mean)
df_interpolated = df_noisy.interpolate(method='linear')
print("Evaluating with Interpolation (After Adding Noise)")
evaluate_classification_model(df_interpolated)
