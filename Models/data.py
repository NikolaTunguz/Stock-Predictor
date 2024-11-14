import pandas as pd

class DataPreprocessing():
    def __init__(self, file):
        self.data = None
        self.file = file
        self.read_data()

    def read_data(self):
        self.data = pd.read_csv(self.file)

    def preprocessing(self):
        self.data = self.data.drop(columns=["Dividends", "Stock Splits"])
        
        #create target columns which are the current columns shifted up 1.
        self.data["tomorrow_open"] = self.data["Open"].shift(-1)
        self.data["tomorrow_high"] = self.data["High"].shift(-1)
        self.data["tomorrow_low"] = self.data["Low"].shift(-1)
        self.data["tomorrow_close"] = self.data["Close"].shift(-1)
        
        return self.data