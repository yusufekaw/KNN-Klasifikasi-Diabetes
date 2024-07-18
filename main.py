from data.pemrosesan_data import ProsesData

if __name__ == "__main__":
    # Tentukan path ke dataset
    dataset_path = 'data/dataset/diabetes.csv'
    
    # Buat instance dari DataProcessor
    proses_data = ProsesData(dataset_path)
    
    # Muat data
    data = proses_data.load_data()
    
    # Tampilkan data
    if data is not None:
        print(data)

    # Lakukan label encoding secara dinamis pada kolom string
    encoded_data, label_encoders = proses_data.label_encode_columns()
    
    # Tampilkan data setelah encoding
    if encoded_data is not None:
        print("\nData setelah encoding:")
        print(encoded_data)