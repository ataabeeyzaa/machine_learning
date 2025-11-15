import pandas as pd

# Veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

df_bfill = df.copy()
df_bfill = df_bfill.fillna(method="bfill")
print(df_bfill.isnull().sum())
