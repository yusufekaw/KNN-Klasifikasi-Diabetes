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
    correct_predictions = (y_test == y_pred).sum()
    incorrect_predictions = (y_test != y_pred).sum()
    
    return y_test, y_pred, accuracy, correct_predictions, incorrect_predictions

# Melakukan evaluasi untuk setiap kombinasi proporsi dan K
results = {}
for proportion in proportions:
    results[proportion] = {}
    for k in k_values:
        y_test, y_pred, accuracy, correct_predictions, incorrect_predictions = evaluate_knn(X, y, proportion, k)
        results[proportion][k] = (y_test, y_pred, accuracy, correct_predictions, incorrect_predictions)

# Menampilkan hasil
for proportion, result in results.items():
    y_test = result[k_values[0]][0]
    y_pred_7 = result[7][1]
    y_pred_9 = result[9][1]
    accuracy_7 = result[7][2]
    accuracy_9 = result[9][2]
    correct_predictions_7 = result[7][3]
    incorrect_predictions_7 = result[7][4]
    correct_predictions_9 = result[9][3]
    incorrect_predictions_9 = result[9][4]

    # Membuat dataframe
    y_test_pred_df = pd.DataFrame({
        'y_test': y_test,
        'y_pred_7': y_pred_7,
        'y_pred_9': y_pred_9
    })

    # Menyimpan ke file .txt
    filename = f'proporsi_{int(proportion*100)}_log.txt'
    with open(filename, 'w') as f:
        f.write(f'Proportion: {proportion*100:.0f}%\n')
        f.write(f'Accuracy for K=7: {accuracy_7:.4f}\n')
        f.write(f'Correct predictions for K=7: {correct_predictions_7}\n')
        f.write(f'Incorrect predictions for K=7: {incorrect_predictions_7}\n\n')
        f.write(f'Accuracy for K=9: {accuracy_9:.4f}\n')
        f.write(f'Correct predictions for K=9: {correct_predictions_9}\n')
        f.write(f'Incorrect predictions for K=9: {incorrect_predictions_9}\n\n')
        f.write(y_test_pred_df.to_string(index=False))
    
    # Menampilkan hasil di konsol
    print(y_test_pred_df.head(10))
    print(f'Proportion: {proportion*100:.0f}%')
    print(f'Accuracy for K=7: {accuracy_7:.4f}')
    print(f'Correct predictions for K=7: {correct_predictions_7}')
    print(f'Incorrect predictions for K=7: {incorrect_predictions_7}')
    print(f'Accuracy for K=9: {accuracy_9:.4f}')
    print(f'Correct predictions for K=9: {correct_predictions_9}')
    print(f'Incorrect predictions for K=9: {incorrect_predictions_9}')
    print()
