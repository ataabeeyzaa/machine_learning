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
    print(f"Method: {method}, Model Type: {model_type}")
    
    # Titanic veri setini yükle
    file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\archive\\Titanic-Dataset.csv'
    df = pd.read_csv(file_path)
    print("Dataset loaded.")
    
    # Kategorik verileri dönüştürme
    categorical_mappings = {
        'Sex': {'male': 0, 'female': 1},
        'Embarked': {'C': 0, 'Q': 1, 'S': 2}
    }
    for column, mapping in categorical_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    print("Categorical variables transformed.")
    
    # Eksik değer içeren sütunları belirleme
    missing_cols = df.columns[df.isnull().any()]
    print(f"Missing columns: {missing_cols}")

    # Eksik değerleri doldurma yöntemleri
    if method == "silme":
        df = df.dropna()
    elif method in ["ffill", "bfill", "sifirla", "mean"]:
        df[missing_cols] = df[missing_cols].infer_objects()
        if method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "sifirla":
            df = df.fillna(0)
        elif method == "mean":
            df = df.interpolate(method='linear')
    else:
        raise ValueError("Bilinmeyen doldurma yöntemi: " + method)
    print(f"Missing values handled with method: {method}")

    # Özellik ve hedef değişkeni ayırma
    features = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])
    target = df['Survived']
    print("Features and target separated.")

    # Normalizasyon
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Features scaled.")

    # Eğitim ve test verilerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)
    print("Training and test data split.")

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
    print(f"Model {model_type} trained.")

    # Modelin performansını değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Model performance evaluated.")

    # Performans metriklerini dosyaya kaydetme
    metrics = f'Method: {method}\nModel: {model_type}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n'
    
    try:
        output_path = f'C:\\Users\\User\\Desktop\\ML_Proje\\b\\{method}_{model_type}_metrics.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(metrics)
        print(f"Metrics saved to {output_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    # Karışıklık matrisini görselleştirme ve farklı bir dizine kaydetme
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title(f'Karışıklık Matrisi ({method} - {model_type})')
        matrix_path = f'C:\\Users\\User\\Desktop\\ML_Proje\\b\\{method}_{model_type}_cm.png'
        plt.savefig(matrix_path)
        plt.close()
        print(f"Confusion matrix saved to {matrix_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")

if __name__ == "main":
    if len(sys.argv) > 2:
        method = sys.argv[1]
        model_type = sys.argv[2]
        train_model(method, model_type)
    else:
        print("Lütfen eksik doldurma yöntemi ve model türünü belirtin: silme, ffill, bfill, sifirla, mean; logistic_regression, decision_tree, random_forest")