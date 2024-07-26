from data.pemrosesan_data import ProsesData
from algoritma.knn import KNN
import pandas as pd 
from collections import Counter

if __name__ == "__main__":
    # Tentukan path ke dataset
    dataset_path = 'data/dataset/diabetes.csv'
    
    # Buat instance dari pemrosesan dara
    proses_data = ProsesData(dataset_path)
    
    # Memuat dataset
    dataset = proses_data.ambilData()
    print("data mentahan")
    print(dataset)

    # Bagi data menjadi X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = proses_data.Split(test_size=0.3, random_state=42)
    print("X_train:")
    print(X_train)
    print("X_test:")
    print(X_test)
    print("y_train:")
    print(y_train)
    print("y_test:")
    print(y_test)

    knn = KNN()
    K=7

    print("proses menghitung jarak . . .")
    matriks_jarak = []
    for i in range(len(X_test)):
        jarak = []
        for ii in range(len(X_train)):
            hitung_jarak = knn.EuclideanDistance(X_test.iloc[i],X_train.iloc[ii], X_train)
            jarak.append(hitung_jarak)
        matriks_jarak.append(jarak)
    print("menghitung jarak selesai")  

    # Menampilkan hasil
    df_jarak = pd.DataFrame(matriks_jarak, index=X_test.index, columns=X_train.index)
    print("Jarak Euclidean antara semua pasangan baris dalam dataset:")
    print(df_jarak)

    # Menemukan k jarak terdekat untuk setiap baris dalam data pengujian
    indeks_terdekat = df_jarak.apply(lambda row: row.nsmallest(K).index, axis=1)
    jarak_terdekat = df_jarak.apply(lambda row: row.nsmallest(K).values, axis=1)
    # Menampilkan hasil jarak terdekat
    hasil_jarak = []
    for i, index_test in enumerate(X_test.index):
        k_indeks_terdekat = indeks_terdekat[index_test]
        k_jarak_tedekat = jarak_terdekat[index_test]
        for ii in range(K):
            index_train = k_indeks_terdekat[ii]
            jarak = k_jarak_tedekat[ii]
            kelas_train = y_train[index_train]
            hasil_jarak.append({
                'Index Test': index_test,
                'Index Train': index_train,
                'Jarak': jarak,
                'Kelas Train': kelas_train
            })
    hasil_jarak_df = pd.DataFrame(hasil_jarak)
    print("Jarak Terdekat dari Setiap Baris Data Pengujian ke Data Pelatihan:")
    print(hasil_jarak_df)

    # Menentukan kelas baru pada data pengujian berdasarkan kelas mayoritas dari tetangga terdekat
    kelas_prediksi = []
    for index_test in X_test.index:
        kelas_terdekat = hasil_jarak_df[hasil_jarak_df['Index Test'] == index_test]['Kelas Train']
        kelas_mayoritas = Counter(kelas_terdekat).most_common(1)[0][0]
        kelas_prediksi.append({
            'Index Test': index_test,
            'Kelas Prediksi':kelas_mayoritas
        })

    # Kelas prediksi
    kelas_prediksi_df = pd.DataFrame(kelas_prediksi)
    kelas_prediksi_df.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    hasil_prediksi = result = pd.concat([kelas_prediksi_df, y_test], axis=1)
    print("Kelas Baru pada Data Pengujian Berdasarkan Kelas Mayoritas dari Tetangga Terdekat:")
    print(hasil_prediksi)

    prediksi_benar = (hasil_prediksi['Kelas Prediksi'] == hasil_prediksi['Outcome']).sum()
    prediksi_salah = hasil_prediksi.shape[0]-prediksi_benar
    akurasi = (prediksi_benar/hasil_prediksi.shape[0])*100
    print("Prediksi Benar : ",prediksi_benar)
    print("Prediksi salah : ",prediksi_salah)
    print("Akurasi : ",round(akurasi,2),"%")