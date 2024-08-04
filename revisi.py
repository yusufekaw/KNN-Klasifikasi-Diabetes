import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Membaca dataset
dataset_path = 'data/dataset/diabetes.csv'
data = pd.read_csv(dataset_path)

# Misalkan kolom terakhir adalah label
X = data.iloc[:, :-1]  # Fitur
y = data.iloc[:, -1]   # Label

# Definisikan proporsi dan nilai K yang akan digunakan
proportions = [0.9, 0.8, 0.7]
k_values = [7, 9]

# Fungsi untuk melakukan pelatihan dan pengujian
def evaluate_knn(X, y, train_size, k):
    # Membagi dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    # Inisialisasi model KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Melatih model
    knn.fit(X_train, y_train)
    
    # Melakukan prediksi
    y_pred = knn.predict(X_test)
    
    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    return y_test, y_pred, accuracy

# Melakukan evaluasi untuk setiap kombinasi proporsi dan K
results = {}
for proportion in proportions:
    results[proportion] = {}
    for k in k_values:
        y_test, y_pred, accuracy = evaluate_knn(X, y, proportion, k)
        results[proportion][k] = (y_test, y_pred, accuracy)

# Menampilkan hasil
for proportion, result in results.items():
    y_test = result[k_values[0]][0]
    y_pred_7 = result[7][1]
    y_pred_9 = result[9][1]
    accuracy_7 = result[7][2]
    accuracy_9 = result[9][2]

    # Membuat dataframe
    y_test_pred_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred_7': y_pred_7,
        'y_pred_9': y_pred_9
    })

    # Menyimpan ke file .txt
    filename = f'data/dataset/proporsi_{int(proportion*100)}.txt'
    with open(filename, 'w') as f:
        f.write(f'Proportion: {proportion*100:.0f}%\n')
        f.write(f'Accuracy for K=7: {accuracy_7:.4f}\n')
        f.write(f'Accuracy for K=9: {accuracy_9:.4f}\n\n')
        f.write(y_test_pred_df.to_string(index=True))
    
    # Menampilkan hasil di konsol
    print(f'Proportion: {proportion*100:.0f}%')
    print(y_test_pred_df)
    print(f'Accuracy for K=7: {accuracy_7:.4f}')
    print(f'Accuracy for K=9: {accuracy_9:.4f}')
    print()