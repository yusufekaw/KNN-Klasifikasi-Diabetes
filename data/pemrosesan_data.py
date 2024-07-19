import pandas as pd
import random

class ProsesData:
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataset = None

    def ambilData(self):
        # Path ke file dataset
        #path = "dataset/heart.csv"
        # Membaca dataset menggunakan Pandas
        self.dataset = pd.read_csv(self.filepath)
        return self.dataset
        
    #split dataset (training testing)
    def Split(self, test_size=0.2, random_state=None):
        if random_state is not None:
            random.seed(random_state)
        
        # Shuffle the data while retaining the original index
        data_shuffled = self.dataset.sample(frac=1, random_state=random_state)
        
        # Determine the split index
        split_index = int(len(data_shuffled) * (1 - test_size))
        
        # Split the data
        train_data = data_shuffled.iloc[:split_index]
        test_data = data_shuffled.iloc[split_index:]
        
        # Identify target column as the last column
        target_column = self.dataset.columns[-1]
        
        # Separate features (X) and target (y)
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        return X_train, X_test, y_train, y_test