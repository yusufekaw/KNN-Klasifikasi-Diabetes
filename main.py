from data.pemrosesan_data import ProsesData

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
    print("\nX_train:")
    print(X_train)
    print("\nX_test:")
    print(X_test)
    print("\ny_train:")
    print(y_train)
    print("\ny_test:")
    print(y_test)