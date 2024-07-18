import pandas as pd
from sklearn.preprocessing import LabelEncoder

class ProsesData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.filepath)
            return self.data
        except FileNotFoundError:
            print("File tidak ditemukan!")
            return None
        
    def label_encode_columns(self):
        if self.data is not None:
            label_encoders = {}
            for column in self.data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                label_encoders[column] = le
            return self.data, label_encoders
        else:
            print("Data belum dimuat!")
            return None, None