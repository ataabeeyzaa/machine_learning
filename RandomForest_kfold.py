import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
import numpy as np

# Veri setini yükleyin ve hazırlayın
file_path = r'C:\\Users\\User\\Desktop\\ML_Proje\\archive\\Titanic-Dataset.csv'
df = pd.read_csv(file_path)
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

# ROS uygulayın
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# K-Fold Cross Validation
k = 5
kfold = KFold(n_splits=k, random_state=42, shuffle=True)
model = RandomForestClassifier(random_state=42)

# Özellikleri ölçekleyin
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Metrikleri hesaplamak ve kaydetmek için değişkenler
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Her fold için metrikleri dosyaya kaydetme
output_dir = 'C:/Users/User/Desktop/ML_Proje/b'
with open(os.path.join(output_dir, 'ros_kfold_random_forest_fold_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write("Fold, Accuracy, Precision, Recall, F1 Score\n")

    kfold_splits = list(kfold.split(X_resampled))
    for fold, (train_index, test_index) in enumerate(kfold_splits):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

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

        cm = confusion_matrix(y_test, y_pred)


        # Metrikleri dosyaya yazdır
        f.write(f"{fold + 1}, {accuracy:.2f}, {precision:.2f}, {recall:.2f}, {f1:.2f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")

        # Karışıklık Matrisi
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.title(f'Random Forest K-Fold Karışıklık Matrisi - Fold {fold + 1} (ROS)')
        plt.savefig(os.path.join(output_dir, f'ros_kfold_random_forest_conf_matrix_fold_{fold + 1}.png'))
        plt.close()

# Ortalama metrikleri kaydet
metrics = (f'K-Fold Cross Validation Results (ROS, k={k}):\n'
           f'Accuracy: {np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}\n'
           f'Precision: {np.mean(precision_scores):.2f} ± {np.std(precision_scores):.2f}\n'
           f'Recall: {np.mean(recall_scores):.2f} ± {np.std(recall_scores):.2f}\n'
           f'F1 Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}\n')
with open(os.path.join(output_dir, 'ros_kfold_random_forest_metrics.txt'), 'w', encoding='utf-8') as f:
    f.write(metrics)


joblib.dump(scaler, 'C:/Users/User/Desktop/ML_Proje/b/kfold_scaler.pkl')
joblib.dump(model, 'C:/Users/User/Desktop/ML_Proje/b/kfold_random_forest_model.pkl')
joblib.dump(X.columns.tolist(), 'C:/Users/User/Desktop/ML_Proje/b/kfold_feature_names.pkl')
