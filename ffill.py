import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

df_ffill = df.copy()
df_ffill = df_ffill.fillna(method="ffill")
print(df_ffill.isnull().sum())
