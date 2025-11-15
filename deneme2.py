import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri eğitin
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)
logistic_pred = logistic_model.predict(X_test)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Metriği hesaplayan fonksiyon
def calculate_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return cm, acc, prec, recall, f1

# Her model için metrikler
logistic_metrics = calculate_metrics(y_test, logistic_pred)
dt_metrics = calculate_metrics(y_test, dt_pred)
rf_metrics = calculate_metrics(y_test, rf_pred)

# Grafik gösterim fonksiyonu
def show_metrics(metrics, model_name):
    cm, acc, prec, recall, f1 = metrics

    # Yeni pencere
    metrics_window = tk.Toplevel()
    metrics_window.title(f"{model_name} Sonuçları")
    metrics_window.geometry("800x600")

    # Karışıklık matrisi grafiği
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='coolwarm')
    plt.title(f"{model_name} Karışıklık Matrisi")
    plt.colorbar(cax)
    plt.xlabel('Tahmin')
    plt.ylabel('Gerçek')

    canvas = FigureCanvasTkAgg(fig, metrics_window)
    canvas.get_tk_widget().pack()

    # Metrik bilgileri
    info = f"Doğruluk: {acc:.2f}\nDuyarlılık: {recall:.2f}\nÖzgüllük: {prec:.2f}\nF1 Skoru: {f1:.2f}"
    tk.Label(metrics_window, text=info, font=("Arial", 14)).pack()

    # Tahmin Yap butonu (Random Forest için)
    if model_name == "Random Forest":
        tk.Button(metrics_window, text="Tahmin Yap", command=prediction_page, bg="green", fg="white", font=("Arial", 14)).pack()

# Tahmin sayfası
def prediction_page():
    pred_window = tk.Toplevel()
    pred_window.title("Random Forest Tahmin Sayfası")
    pred_window.geometry("400x400")

    tk.Label(pred_window, text="Tahmin Verilerini Girin", font=("Arial", 16)).pack()

    entries = {}
    fields = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_Male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    for field in fields:
        tk.Label(pred_window, text=field, font=("Arial", 12)).pack()
        entry = tk.Entry(pred_window)
        entry.pack()
        entries[field] = entry

    def make_prediction():
        user_data = np.array([[float(entries[field].get()) for field in fields]])
        pred = rf_model.predict(user_data)[0]
        prob = rf_model.predict_proba(user_data)[0][1]
        result = "Hayatta Kalacak" if pred == 1 else "Hayatta Kalamayacak"
        messagebox.showinfo("Tahmin", f"Sonuç: {result}\nOlasılık: {prob:.2f}")

    tk.Button(pred_window, text="Tahmin Yap", command=make_prediction, bg="blue", fg="white", font=("Arial", 14)).pack()

# Ana Sayfa
def main_page():
    root = tk.Tk()
    root.title("Titanic Veri Seti")
    root.geometry("800x400")

    tk.Label(root, text="Titanic Veri Seti", font=("Arial", 20), bg="orange", fg="white").pack(fill=tk.X)

    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)

    buttons = ["Ham Veri", "", "", "", ""]
    colors = ["red", "green", "blue", "yellow", "purple"]

    for i, text in enumerate(buttons):
        button = tk.Button(frame, text=text, bg=colors[i], font=("Arial", 16), width=15, height=5)
        if text == "Ham Veri":
            button.config(command=lambda: model_selection_page())
        button.grid(row=0, column=i, sticky="nsew")

    root.mainloop()

# Model seçimi sayfası
def model_selection_page():
    selection_window = tk.Toplevel()
    selection_window.title("Model Seçimi")
    selection_window.geometry("800x400")

    tk.Label(selection_window, text="Bir Model Seçin", font=("Arial", 18), bg="cyan").pack(fill=tk.X)

    frame = tk.Frame(selection_window)
    frame.pack(fill=tk.BOTH, expand=True)

    models = ["Logistic Regression", "Karar Ağaçları", "Random Forest"]
    metrics = [logistic_metrics, dt_metrics, rf_metrics]
    colors = ["pink", "lime", "skyblue"]

    for i, model in enumerate(models):
        button = tk.Button(frame, text=model, bg=colors[i], font=("Arial", 16), width=20, height=5)
        button.config(command=lambda m=metrics[i], name=model: show_metrics(m, name))
        button.grid(row=0, column=i, sticky="nsew")

main_page()
