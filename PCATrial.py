import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Örnek veri setini oluştur
data = {
    'Dil': ['Python', 'Java', 'C++', 'JavaScript', 'Ruby'],
    'Negatif': [10, 8, 15, 12, 9],
    'Weakly Negatif': [5, 6, 10, 7, 4],
    'Nötr': [30, 25, 20, 28, 32],
    'Weakly Pozitif': [15, 20, 10, 18, 16],
    'Pozitif': [40, 35, 25, 38, 42]
}

df = pd.DataFrame(data)

# Dil sütununu indeks olarak ayarla
df.set_index('Dil', inplace=True)

# Verileri standartlaştır
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA uygula
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# İki bileşeni görselleştir
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel('Bileşen 1')
plt.ylabel('Bileşen 2')

# Dil etiketlerini ekle
for i, dil in enumerate(df.index):
    plt.annotate(dil, (reduced_data[i, 0], reduced_data[i, 1]))

plt.title('PCA Analizi Sonucu')
plt.grid()
plt.show()
