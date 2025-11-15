import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Gürültü eklenmiş Titanic veri setini yükle
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\b\\Titanic-Dataset-noisy.csv'
df = pd.read_csv(file_path)

# Kategorik Verileri Dönüştürme
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Sayısal ve kategorik sütunları ayır
numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df.select_dtypes(include=[object]).columns.tolist()

# Eksik Değerleri Doldurma
# Sayısal sütunlar için medyan stratejisi
num_imputer = SimpleImputer(strategy='median')
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

# Kategorik sütunlar için en sık görülen değer (mode) stratejisi
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

# Uç Değer Analizi ve İşleme (Opsiyonel)
def remove_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

numerical_columns.remove('Survived')  # Hedef sütunu çıkar
df = remove_outliers(df, numerical_columns)

# Özellik Normalizasyonu
scaler = StandardScaler()
features = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
target = df['Survived']
features_scaled = scaler.fit_transform(features)

# K-Fold Cross Validation
k = 10
kfold = KFold(n_splits=k, random_state=42, shuffle=True)
model = LogisticRegression(random_state=42)

# Metrikleri hesaplamak ve kaydetmek için değişkenler
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

print("Fold, Accuracy, Precision, Recall, F1 Score")
kfold_splits = list(kfold.split(features_scaled))
for fold, (train_index, test_index) in enumerate(kfold_splits):
    X_train, X_test = features_scaled[train_index], features_scaled[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performans Metrikleri
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Performans Metriklerini Yazdır
    print(f"{fold + 1}, {accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}")

    # Karışıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'Logistic Regression K-Fold Karışıklık Matrisi - Fold {fold + 1}')
    plt.savefig(f'C:\\Users\\User\\Desktop\\ML_Proje\\b\\kfold_logistic_regression_conf_matrix_fold_{fold + 1}.png')
    plt.close()

# Ortalama metrikleri hesapla ve yazdır
print("\nOrtalama Metrikler")
print(f"Accuracy: {np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}")
print(f"Precision: {np.mean(precision_scores):.2f} ± {np.std(precision_scores):.2f}")
print(f"Recall: {np.mean(recall_scores):.2f} ± {np.std(recall_scores):.2f}")
print(f"F1 Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
