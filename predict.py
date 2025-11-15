import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

# Komut satırından gelen verileri al
print("Başlangıç", file=sys.stderr)
if len(sys.argv[1:]) != 10:
    print(f"Hatalı veya eksik veri girdisi. Beklenen 10 argüman, ancak {len(sys.argv[1:])} argüman alındı: {sys.argv[1:]}", file=sys.stderr)
    sys.exit(1)

# Yöntem ve kullanıcıdan gelen verileri alın
method = sys.argv[1]
try:
    pclass, age, sibsp, parch, fare, sex_male, embarked_c, embarked_q, embarked_s = map(float, sys.argv[2:])
except ValueError as e:
    print(f"Değer hatası: {e}. Alınan veriler: {sys.argv[2:]}", file=sys.stderr)
    sys.exit(1)

# Debug: Alınan argümanları yazdır
print(f"Alınan argümanlar: {method}, {pclass}, {age}, {sibsp}, {parch}, {fare}, {sex_male}, {embarked_c}, {embarked_q}, {embarked_s}", file=sys.stderr)

# Veri setini yükleyin ve hazırlayın
print("Veri seti yükleniyor", file=sys.stderr)
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("Veri seti hazır", file=sys.stderr)

X = df.drop(columns=['Survived'])
y = df['Survived']

# En iyi doldurma stratejisini kullanın (Median Fill)
X = X.fillna(X.median())

# Kullanıcıdan gelen veriyi aynı şekilde kodlayın
input_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_c, embarked_q, embarked_s]])

# Debug: Kodlanmış kullanıcı verisini yazdır
print(f"Kodlanmış kullanıcı verisi: {input_data}", file=sys.stderr)

# Modeli eğitmek ve tahmin yapmak için metot seçimi
print(f"Seçilen yöntem: {method}", file=sys.stderr)
if method == "ham":
    # Ham veri için en iyi modeli kullan
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    print("Model ham veri ile eğitildi", file=sys.stderr)
else:
    # ROS, BOS veya SMOTE ile veri dengesini sağlayın
    if method == "ros":
        resampler = RandomOverSampler(random_state=42)
    elif method == "bos":
        resampler = RandomUnderSampler(random_state=42)
    elif method == "smote":
        resampler = SMOTE(random_state=42)
    else:
        raise ValueError("Geçersiz metot seçimi. 'ros', 'bos', 'smote' veya 'ham' olmalıdır.")
    
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    print("Model dengeli veri ile eğitildi", file=sys.stderr)

# Tahmin yapın
try:
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    print(f"Tahmin: {prediction[0]}, Olasılık: {probability[0][1]}", file=sys.stderr)
    # Tahmin ve olasılığı yazdırın
    print(int(prediction[0]))  # Tahmin sonucunu tamsayıya dönüştürerek yazdırın
    print(probability[0][1])  # Hayatta kalma olasılığını yazdırın
except Exception as e:
    print(f"Tahmin hatası: {e}", file=sys.stderr)
    sys.exit(1)
