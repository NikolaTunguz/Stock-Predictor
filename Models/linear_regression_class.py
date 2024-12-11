from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
import pandas as pd

class MyLinearRegression:
    def __init__(self, X, y):
        #core model variables
        self.model = LinearRegression()
        self.X = X
        self.y = y

        #model varables for train/val/test split
        self.X_train = None
        self.y_train = None

        self.X_val = None
        self.y_val = None

        self.X_test = None
        self.y_test = None

        self.split_data()
    
    #function to split data into training and test
    def split_data(self):
        temp_X = None
        temp_y = None
        self.X_train, temp_X, self.y_train, temp_y = train_test_split(self.X, self.y, test_size = 0.2, shuffle = False)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(temp_X, temp_y, test_size = 0.5, shuffle = False)

    #function to train the model
    def train(self):
        self.model.fit(self.X_train, self.y_train)

    #private function to predict on test dataset
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction

    #function to evaluate prediction performance
    def evaluate(self):
        prediction = self.predict(self.X_test)
        percent_error = mean_absolute_percentage_error(self.y_test, prediction)
        root_mean_error = root_mean_squared_error(self.y_test, prediction)
        return percent_error, root_mean_error
    
    def get_split_MAPE(self):
        prediction = self.predict(self.X_train)
        train_acc = mean_absolute_percentage_error(self.y_train, prediction)

        prediction = self.predict(self.X_val)
        val_acc = mean_absolute_percentage_error(self.y_val, prediction)

        prediction = self.predict(self.X_test)
        test_acc = mean_absolute_percentage_error(self.y_test, prediction)

        values = [train_acc, val_acc, test_acc]
        return values
    
    def get_split_RMSE(self):
        prediction = self.predict(self.X_train)
        train_acc = root_mean_squared_error(self.y_train, prediction)

        prediction = self.predict(self.X_val)
        val_acc = root_mean_squared_error(self.y_val, prediction)

        prediction = self.predict(self.X_test)
        test_acc = root_mean_squared_error(self.y_test, prediction)

        values = [train_acc, val_acc, test_acc]
        return values

    def predict_ahead(self, days):
        predictions = []
        current_day = self.X_test.iloc[-1].values.reshape(1, -1) 
        current_day = pd.DataFrame(current_day, columns = self.X.columns)  

        for i in range(days):
            
            output = self.model.predict(current_day)
            
            predicted_values = output[0]  
            
            predictions.append(f"{i + 1} Day Prediction: Open = {predicted_values[0]}, High = {predicted_values[1]}, Low = {predicted_values[2]}, Close = {predicted_values[3]}")

            current_day = pd.DataFrame([predicted_values], columns=self.X.columns)  
        
        output = "\n".join(predictions)
        return output
        
            
