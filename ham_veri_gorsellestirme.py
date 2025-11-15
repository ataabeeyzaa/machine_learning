import pandas as pd
import matplotlib.pyplot as plt

# Veri setini yükleyin ve hazırlayın
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')
df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Ham veriyi görselleştirin
plt.scatter(df['Age'], df['Fare'], color='blue')
plt.xlabel('Yaş')
plt.ylabel('Bilet Ücreti')
plt.title('Ham Veri Görselleştirme')
plt.savefig('C:/Users/User/Desktop/ML_Proje/b/original_image.png')
plt.close()
