import pandas as pd

# Titanic veri setini yükleyin
file_path = 'C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv'
df = pd.read_csv(file_path)

# Hayatta kalan ve ölenlerin sayısını bulma
survived_count = df[df['Survived'] == 1].shape[0]
dead_count = df[df['Survived'] == 0].shape[0]

# Sonuçları yazdır
print(f"Hayatta Kalan: {survived_count}")
print(f"Hayatta Kalamayan: {dead_count}")
