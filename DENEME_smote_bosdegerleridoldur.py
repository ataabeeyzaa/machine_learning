import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Veri setini yükleyin ve hazırlayın
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

# Eksik değerleri doldurmak için farklı stratejiler
strategies = {
    'zero_fill': X.fillna(0),
    'ffill': X.ffill(),
    'bfill': X.bfill(),
    'mean_fill': X.fillna(X.mean()),
    'median_fill': X.fillna(X.median())
}

results = []

# ROS uygulayıp, modeli eğitip, performans metriklerini değerlendirin
for strategy_name, X_filled in strategies.items():
    # ROS uygulayın
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_filled, y)

    # Veri setini ayırın
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Modeli eğitin
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Tahmin yapın
    y_pred = model.predict(X_test)
    
    # Performans metriklerini değerlendirin
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Sonuçları kaydedin
    results.append({
        'strategy': strategy_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Sonuçları DataFrame olarak gösterin
results_df = pd.DataFrame(results)
print(results_df)

# Performans Metriklerini Görselleştir
plt.figure(figsize=(12, 8))
sns.barplot(x='strategy', y='accuracy', data=results_df, color='b', label='Accuracy')
sns.barplot(x='strategy', y='precision', data=results_df, color='g', label='Precision')
sns.barplot(x='strategy', y='recall', data=results_df, color='r', label='Recall')
sns.barplot(x='strategy', y='f1_score', data=results_df, color='c', label='F1 Score')
plt.title('Doldurma Stratejilerinin Performans Metrikleri (ROS)')
plt.xlabel('Doldurma Stratejisi')
plt.ylabel('Değer')
plt.legend(loc='upper left')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/ros_fill_strategies_performance.png')
plt.show()
