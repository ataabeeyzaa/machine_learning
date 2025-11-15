import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri işleme
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.fillna(0, inplace=True)
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Modeli
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Performans
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Model Doğruluğu: {accuracy_rf}")

# Tahmin Fonksiyonu
def predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_c, embarked_q, embarked_s):
    user_data = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_c, embarked_q, embarked_s]])
    prediction = random_forest.predict(user_data)
    probability = random_forest.predict_proba(user_data)[0][1]
    return prediction[0], probability

# Tkinter GUI
def predict_button_clicked():
    try:
        pclass = int(entry_pclass.get())
        age = float(entry_age.get())
        sibsp = int(entry_sibsp.get())
        parch = int(entry_parch.get())
        fare = float(entry_fare.get())
        sex = var_sex.get()
        embarked = var_embarked.get()

        sex_male = 1 if sex == "Male" else 0
        embarked_c = 1 if embarked == "C" else 0
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        prediction, probability = predict_survival(pclass, age, sibsp, parch, fare, sex_male, embarked_c, embarked_q, embarked_s)

        result_text = "Hayatta Kalacak" if prediction == 1 else "Hayatta Kalamayacak"
        messagebox.showinfo("Tahmin Sonucu", f"{result_text}\nHayatta Kalma Olasılığı: {probability:.2f}")
    except Exception as e:
        messagebox.showerror("Hata", f"Girdi hatası: {str(e)}")

# Tkinter Arayüz Tasarımı
root = tk.Tk()
root.title("Titanic Hayatta Kalma Tahmini")
root.geometry("400x400")

tk.Label(root, text="Yolcu Sınıfı (1, 2, 3):").pack()
entry_pclass = tk.Entry(root)
entry_pclass.pack()

tk.Label(root, text="Yaş:").pack()
entry_age = tk.Entry(root)
entry_age.pack()

tk.Label(root, text="Kardeş/Eş Sayısı:").pack()
entry_sibsp = tk.Entry(root)
entry_sibsp.pack()

tk.Label(root, text="Ebeveyn/Çocuk Sayısı:").pack()
entry_parch = tk.Entry(root)
entry_parch.pack()

tk.Label(root, text="Bilet Ücreti:").pack()
entry_fare = tk.Entry(root)
entry_fare.pack()

tk.Label(root, text="Cinsiyet:").pack()
var_sex = tk.StringVar(value="Male")
tk.Radiobutton(root, text="Erkek", variable=var_sex, value="Male").pack()
tk.Radiobutton(root, text="Kadın", variable=var_sex, value="Female").pack()

tk.Label(root, text="Biniş Noktası:").pack()
var_embarked = tk.StringVar(value="S")
tk.Radiobutton(root, text="Southampton (S)", variable=var_embarked, value="S").pack()
tk.Radiobutton(root, text="Cherbourg (C)", variable=var_embarked, value="C").pack()
tk.Radiobutton(root, text="Queenstown (Q)", variable=var_embarked, value="Q").pack()

tk.Button(root, text="Tahmin Et", command=predict_button_clicked).pack()

root.mainloop()
