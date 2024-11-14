import pandas as pd

class DataPreprocessing():
    def __init__(self, file):
        self.data = None
        self.file = file

    def read_data(self):
        self.data = pd.read_csv(self.file)
        return self.data

    def preprocessing(self):
        self.data = self.data.drop(columns=["Dividends", "Stock Splits"])
        print(self.data.head())
        
        #create target columns which are the current columns shifted up 1.
        self.data["tommorow_open"] = self.data["Open"].shift(-1)
        self.data["tommorow_high"] = self.data["High"].shift(-1)
        self.data["tommorow_low"] = self.data["Low"].shift(-1)
        self.data["tommorow_close"] = self.data["Close"].shift(-1)
        
        

        #drop rows with a null value in any column.
        self.data = self.data.dropna()

        return self.data