#nn imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#data imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, features, targets):
        super(NeuralNetwork, self).__init__()
        #define global variables 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_val_predictions = []
        self.epoch_val_actual = []

        self.epoch_train_predictions = []
        self.epoch_train_actual = []
        self.epoch_train_mape = []
        self.epoch_val_mape = []
        self.epoch_train_rmse = []
        self.epoch_val_rmse = []

        self.num_epochs = None
        self.features = features
        self.targets = targets

        #define model
        hidden_size1 = 512
        hidden_size2 = 256
        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size2, output_dim)
        
    
    def forward(self, x):
        #forward pass through model
        output = self.fc1(x)
        output = self.activation1(output)

        output = self.fc2(output)
        output = self.activation2(output)
        
        output = self.fc3(output)
        
        return output
        

    def prepare_data(self):
        #split dataset
        #not shuffled so that the data maintains chronological order
        self.x_train, x_temp, self.y_train, y_temp = train_test_split(self.features.values, self.targets.values, test_size=0.2, shuffle=False) 
        self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=False)

        #convert to tensors
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.x_val = torch.tensor(self.x_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

        #place datasets in dataloader
        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        test_dataset = TensorDataset(self.x_test, self.y_test)
        batch_size = 64
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    def train_model(self):
        self.prepare_data()
        #hyperparameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        num_epochs = 100

        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            self.train()
            batch_train_loss = []
            batch_val_loss = []
            #MAPE & RMSE
            batch_val_predictions = []
            batch_val_actual = []
            batch_train_predictions = []
            batch_train_actual = []

            #training
            for(features, targets) in self.train_loader:
                features,targets = features.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                predictions = self(features)
                loss = criterion(predictions, targets)
                batch_train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

                batch_train_predictions.append(predictions.detach().cpu().numpy())
                batch_train_actual.append(targets.detach().cpu().numpy())

            #epoch training loss
            train_loss = sum(batch_train_loss) / len(batch_train_loss)
            self.epoch_train_loss.append(train_loss)

            #flatten batch for input to MAPE & RMSE function
            self.epoch_train_predictions = np.concatenate(batch_train_predictions, axis=0)
            self.epoch_train_actual = np.concatenate(batch_train_actual, axis=0)

            #calculate MAPE & RMSE on the training data
            self.epoch_train_mape.append(mean_absolute_percentage_error(self.epoch_train_actual, self.epoch_train_predictions))
            self.epoch_train_rmse.append(root_mean_squared_error(self.epoch_train_actual, self.epoch_train_predictions))

            #validation
            self.eval()
            with torch.no_grad():
                for(features,targets) in self.val_loader:
                    features,targets = features.to(self.device), targets.to(self.device)

                    predictions = self(features)
                    loss = criterion(predictions, targets)
                    batch_val_loss.append(loss.item())

                    batch_val_predictions.append(predictions.cpu().numpy())
                    batch_val_actual.append(targets.cpu().numpy())        

                #flatten batch for input to MAPE & RMSE function
                self.epoch_val_predictions = np.concatenate(batch_val_predictions, axis=0)
                self.epoch_val_actual = np.concatenate(batch_val_actual, axis=0) 

                # Calculate MAPE&RMSE on the validation data
                self.epoch_val_mape.append(mean_absolute_percentage_error(self.epoch_val_actual, self.epoch_val_predictions))
                self.epoch_val_rmse.append(root_mean_squared_error(self.epoch_val_actual, self.epoch_val_predictions))

                #epoch validation loss
                val_loss = sum(batch_val_loss) / len(batch_val_loss)
                self.epoch_val_loss.append(val_loss)
            #print(f"Epoch: {epoch+1} train loss: {self.epoch_train_loss[epoch]:.4f} val loss: {self.epoch_val_loss[epoch]:.4f} val MAPE: {val_mape:.4f}")
  
    #graph training and validation loss
    def graph(self):
        plt.figure(figsize=(12, 6))
        epochs = range(1, self.num_epochs+1)

        #plotting loss
        plt.subplot(3,1,1)

        plt.title("Neural Network Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, self.epoch_train_loss, label="Training Loss")
        plt.plot(epochs, self.epoch_val_loss, label="Validation Loss")
        plt.legend()
        
        #plotting mape
        plt.subplot(3,1,2)
        plt.title("Neural Network MAPE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Absolute Percent Error")
        plt.plot(epochs, self.epoch_train_mape, label="Training MAPE")
        plt.plot(epochs, self.epoch_val_mape, label="Validation MAPE")
        plt.legend()
        
        #plotting rmse
        plt.subplot(3,1,3)
        plt.title("Neural Network RMSE")
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")
        plt.plot(epochs, self.epoch_train_rmse, label="Training RMSE")
        plt.plot(epochs, self.epoch_val_rmse, label="Validation RMSE")
        plt.legend()

        plt.tight_layout()
        plt.show()

    #generic prediction function for evaluation
    def predict(self, data):
        batch_predictions = []
        with torch.no_grad():
            for(features,targets) in data:
                features,targets = features.to(self.device), targets.to(self.device)
                prediction = self(features)
                batch_predictions.append(prediction.cpu().numpy())
        
        return np.concatenate(batch_predictions, axis=0)

    def evaluate(self):
        prediction = self.predict(self.test_loader)
        percent_error = mean_absolute_percentage_error(self.y_test, prediction)
        root_mean_error = root_mean_squared_error(self.y_test, prediction)
        return percent_error, root_mean_error

    #next n days prediction
    def predict_ahead(self, days):
        self.eval()
        predictions = []
        predictions_string = []

        with torch.no_grad():
            self.most_recent_day = self.targets.tail(1).values
            self.most_recent_day = torch.tensor(self.most_recent_day, dtype=torch.float32).to(self.device)

            #forward pass on the most recent day
            for i in range(days):
                prediction = self(self.most_recent_day)
                predictions.append(prediction.cpu().numpy().flatten())

                predicted_values = predictions[0]  
                predictions_string.append(f"{i + 1} Day Prediction: Open = {predicted_values[0]}, High = {predicted_values[1]}, Low = {predicted_values[2]}, Close = {predicted_values[3]}")

                #update input for the next day.
                self.most_recent_day = prediction

        #return predictions
    
        output = "\n".join(predictions_string)
        return output
    

    def print_results(self, days):
        daily_prices = self.predict_ahead(days)

        for i in range(len(daily_prices)):
            print("Predicted Prices (Open  High  Low  Close):", 
            daily_prices[i][0], " | ", daily_prices[i][1], " | ", 
            daily_prices[i][2], " | ", daily_prices[i][3])


    def get_split_MAPE(self):
        prediction = self.predict(self.train_loader)
        train_acc = mean_absolute_percentage_error(self.y_train, prediction)

        prediction = self.predict(self.val_loader)
        val_acc = mean_absolute_percentage_error(self.y_val, prediction)

        prediction = self.predict(self.test_loader)
        test_acc = mean_absolute_percentage_error(self.y_test, prediction)

        values = [train_acc, val_acc, test_acc]
        return values
    
    def get_split_RMSE(self):
        prediction = self.predict(self.train_loader)
        train_acc = root_mean_squared_error(self.y_train, prediction)

        prediction = self.predict(self.val_loader)
        val_acc = root_mean_squared_error(self.y_val, prediction)

        prediction = self.predict(self.test_loader)
        test_acc = root_mean_squared_error(self.y_test, prediction)

        values = [train_acc, val_acc, test_acc]
        return values