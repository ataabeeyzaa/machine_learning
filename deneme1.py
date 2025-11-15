import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Veri setini yükleme ve ön işleme
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımlama
logistic_model = LogisticRegression(max_iter=1000)
decision_tree_model = DecisionTreeClassifier(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)

models = {
    "Lojistik Regresyon": logistic_model,
    "Karar Ağacı": decision_tree_model,
    "Random Forest": random_forest_model
}

# Model sonuçlarını hesaplama
model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "Doğruluk": accuracy_score(y_test, y_pred),
        "Karışıklık Matrisi": confusion_matrix(y_test, y_pred),
        "Duyarlılık": recall_score(y_test, y_pred),
        "Özgüllük": precision_score(y_test, y_pred),
        "F1 Skoru": f1_score(y_test, y_pred)
    }
    model_results[name] = metrics

# Grafik oluşturma fonksiyonu
def plot_metrics(metrics, model_name):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{model_name} Performans Metrikleri", fontsize=16, color='blue')

    # Karışıklık Matrisi
    cm = metrics["Karışıklık Matrisi"]
    axs[0, 0].imshow(cm, cmap='Blues')
    axs[0, 0].set_title("Karışıklık Matrisi", fontsize=12)
    axs[0, 0].set_xlabel("Tahmin", fontsize=10)
    axs[0, 0].set_ylabel("Gerçek", fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[0, 0].text(j, i, cm[i, j], ha='center', va='center', color='red')

    # Diğer metrikler
    axs[0, 1].bar(metrics.keys(), [metrics[k] for k in metrics if k != "Karışıklık Matrisi"], color='skyblue')
    axs[0, 1].set_title("Metrikler", fontsize=12)
    axs[0, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Tahmin fonksiyonu
def predict_survival(model, inputs):
    inputs_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs_array)
    probability = model.predict_proba(inputs_array)[0][1]
    return prediction[0], probability

# Tkinter arayüzü
def show_metrics(model_name):
    metrics = model_results[model_name]
    fig = plot_metrics(metrics, model_name)

    new_window = tk.Toplevel(root)
    new_window.title(f"{model_name} Sonuçları")
    canvas = FigureCanvasTkAgg(fig, master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def show_random_forest_prediction():
    def make_prediction():
        inputs = [
            int(pclass_entry.get()),
            float(age_entry.get()),
            int(sibsp_entry.get()),
            int(parch_entry.get()),
            float(fare_entry.get()),
            int(sex_male_var.get()),
            int(embarked_c_var.get()),
            int(embarked_q_var.get()),
            int(embarked_s_var.get())
        ]
        prediction, probability = predict_survival(random_forest_model, inputs)
        result_label.config(text=f"Tahmin: {'Hayatta Kalacak' if prediction == 1 else 'Hayatta Kalamayacak'}\nOlasılık: {probability:.2f}")

    prediction_window = tk.Toplevel(root)
    prediction_window.title("Random Forest Tahmin")

    tk.Label(prediction_window, text="Pclass:").grid(row=0, column=0)
    pclass_entry = tk.Entry(prediction_window)
    pclass_entry.grid(row=0, column=1)

    tk.Label(prediction_window, text="Age:").grid(row=1, column=0)
    age_entry = tk.Entry(prediction_window)
    age_entry.grid(row=1, column=1)

    tk.Label(prediction_window, text="SibSp:").grid(row=2, column=0)
    sibsp_entry = tk.Entry(prediction_window)
    sibsp_entry.grid(row=2, column=1)

    tk.Label(prediction_window, text="Parch:").grid(row=3, column=0)
    parch_entry = tk.Entry(prediction_window)
    parch_entry.grid(row=3, column=1)

    tk.Label(prediction_window, text="Fare:").grid(row=4, column=0)
    fare_entry = tk.Entry(prediction_window)
    fare_entry.grid(row=4, column=1)

    sex_male_var = tk.IntVar()
    tk.Checkbutton(prediction_window, text="Male", variable=sex_male_var).grid(row=5, column=0, columnspan=2)

    embarked_c_var = tk.IntVar()
    tk.Checkbutton(prediction_window, text="Embarked C", variable=embarked_c_var).grid(row=6, column=0, columnspan=2)

    embarked_q_var = tk.IntVar()
    tk.Checkbutton(prediction_window, text="Embarked Q", variable=embarked_q_var).grid(row=7, column=0, columnspan=2)

    embarked_s_var = tk.IntVar()
    tk.Checkbutton(prediction_window, text="Embarked S", variable=embarked_s_var).grid(row=8, column=0, columnspan=2)

    tk.Button(prediction_window, text="Tahmin Yap", command=make_prediction).grid(row=9, column=0, columnspan=2)

    result_label = tk.Label(prediction_window, text="")
    result_label.grid(row=10, column=0, columnspan=2)

root = tk.Tk()
root.title("Model Performansı")

frame = tk.Frame(root, bg='lightblue')
frame.pack(padx=20, pady=20)

tk.Label(frame, text="Ham Veri", font=("Arial", 16), bg='lightblue').pack()

tk.Button(frame, text="Lojistik Regresyon", command=lambda: show_metrics("Lojistik Regresyon"), bg='skyblue', fg='white').pack(fill='x', pady=5)

tk.Button(frame, text="Karar Ağacı", command=lambda: show_metrics("Karar Ağacı"), bg='green', fg='white').pack(fill='x', pady=5)

tk.Button(frame, text="Random Forest", command=lambda: show_metrics("Random Forest"), bg='orange', fg='white').pack(fill='x', pady=5)

# Random Forest Tahmin Ekle
frame_rf = tk.Frame(root, bg='lightblue')
frame_rf.pack(padx=20, pady=20)

tk.Label(frame_rf, text="Random Forest Tahmin", font=("Arial", 16), bg='lightblue').pack()

tk.Button(frame_rf, text="Tahmin Sayfası", command=show_random_forest_prediction, bg='red', fg='white').pack(fill='x', pady=5)

root.mainloop()
