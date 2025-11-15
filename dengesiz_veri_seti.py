import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleyin
df = pd.read_csv('C:/Users/User/Desktop/ML_Proje/archive/Titanic-Dataset.csv')

# Sınıf dağılımını kontrol edin
class_distribution = df['Survived'].value_counts()

print(class_distribution)

# Sınıf dağılımını görselleştirin
sns.countplot(x='Survived', data=df)
plt.title('Sınıf Dağılımı')
plt.show()
