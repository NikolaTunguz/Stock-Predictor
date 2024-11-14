from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

class LinearRegression:
    def __init__(self, X, y):
        self.model = LinearRegression()
        self.X = X
        self.y = y

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.split_data()
    
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        prediction = self.model.predict(self.X_test)
        return prediction

    def evaluate(self):
        prediction = self.predict()
        accuracy = mean_absolute_percentage_error(self.y_test, prediction)
        return accuracy
