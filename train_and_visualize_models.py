import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi oku
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Eksik verileri doldurma seçenekleri
def fill_data(method):
    if method == "dropna":
        df_filled = df.dropna()
    elif method == "ffill":
        df_filled = df.fillna(method="ffill")
    elif method == "bfill":
        df_filled = df.fillna(method="bfill")
    elif method == "zero":
        df_filled = df.fillna(0)
    elif method == "interpolate":
        df_filled = df.interpolate(method="linear")
    return df_filled

# Modeli eğit ve test et
def train_and_evaluate(model_type, method):
    df_filled = fill_data(method)
    X = df_filled.drop('Survived', axis=1)
    y = df_filled['Survived']

    # Veriyi eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model seçimi
    if model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "knn":
        model = KNeighborsClassifier()

    # Modeli eğit
    model.fit(X_train, y_train)

    # Tahmin yap
    y_pred = model.predict(X_test)

    # Sonuçları döndür
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return cm, report

# Görselleştirme (confusion matrix)
def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Modeli seç ve metrikleri döndür
model_type = 'logistic_regression'  # Bu değeri C# tarafından belirleyeceğiz
method = 'ffill'  # Bu değeri de C# tarafından belirleyeceğiz
cm, report = train_and_evaluate(model_type, method)
plot_confusion_matrix(cm)
print(report)
