import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import sys

def train_model(method, model_type):
    # Titanic veri setini yükle
    file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\Titanic-Dataset.csv'
    df = pd.read_csv(file_path)

    # Kategorik verileri dönüştürme
    categorical_mappings = {
        'Sex': {'male': 0, 'female': 1},
        'Embarked': {'C': 0, 'Q': 1, 'S': 2}
    }
    for column, mapping in categorical_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)

    # Eksik değerleri doldurma yöntemleri
    if method == "silme":
        df = df.dropna()
    elif method == "ffill":
        df = df.ffill()
    elif method == "bfill":
        df = df.bfill()
    elif method == "sifirla":
        df = df.fillna(0)
    elif method == "mean":
        df = df.infer_objects(copy=False).interpolate(method='linear')
    else:
        raise ValueError("Bilinmeyen doldurma yöntemi: " + method)

    # Özellik ve hedef değişkeni ayırma
    features = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
    target = df['Survived']

    # Normalizasyon
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Eğitim ve test verilerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

    # Model seçimi ve eğitimi
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Bilinmeyen model türü: " + model_type)
    
    model.fit(X_train, y_train)

    # Modelin performansını değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Metrikleri dosyaya kaydetme
    metrics = f'Method: {method}\nModel: {model_type}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n'
    with open(f'C:\\Users\\User\\Desktop\\ML_Proje\\b\\{method}_{model_type}_metrics.txt', 'w', encoding='utf-8') as f:
        f.write(metrics)

    # Karışıklık matrisi görselleştirme
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'Karışıklık Matrisi ({method} - {model_type})')
    matrix_path = f'C:\\Users\\User\\Desktop\\ML_Proje\\b\\{method}_{model_type}_cm.png'
    plt.savefig(matrix_path)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        method = sys.argv[1]
        model_type = sys.argv[2]
        train_model(method, model_type)
    else:
        print("Lütfen eksik doldurma yöntemi ve model türünü belirtin: silme, ffill, bfill, sifirla, mean; logistic_regression, decision_tree, random_forest")
