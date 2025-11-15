import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import os

# Veri setini yükleyin ve hazırlayın
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

# Yöntemleri tanımla
methods = {
    'ROS': RandomOverSampler(random_state=42),
    'BOS': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42)
}

# Sonuçları saklamak için
results = {}

# Her yöntem için modeli eğit ve değerlendir
for method_name, method in methods.items():
    X_resampled, y_resampled = method.fit_resample(X, y)
    
    # Dengesiz veri setiyle eğitim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_unbalanced = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Dengeli veri setiyle eğitim
    X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model_resampled = RandomForestClassifier(random_state=42)
    model_resampled.fit(X_train_resampled, y_train_resampled)
    y_pred_resampled = model_resampled.predict(X_test_resampled)
    metrics_balanced = {
        'accuracy': accuracy_score(y_test_resampled, y_pred_resampled),
        'precision': precision_score(y_test_resampled, y_pred_resampled),
        'recall': recall_score(y_test_resampled, y_pred_resampled),
        'f1_score': f1_score(y_test_resampled, y_pred_resampled)
    }

    # Sonuçları sakla
    results[method_name] = {
        'unbalanced': metrics_unbalanced,
        'balanced': metrics_balanced
    }

# Sonuçları yazdır
for method_name, metrics in results.items():
    print(f'\nMethod: {method_name}')
    print('Dengesiz Veri Seti:')
    for metric, value in metrics['unbalanced'].items():
        print(f'  {metric}: {value:.4f}')
    print('Dengeli Veri Seti:')
    for metric, value in metrics['balanced'].items():
        print(f'  {metric}: {value:.4f}')
