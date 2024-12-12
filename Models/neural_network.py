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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        num_epochs = 20

        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            self.train()
            batch_train_loss = []
            batch_val_loss = []
            #MAPE
            batch_val_predictions = []
            batch_val_actual = []

            #training
            for(features, targets) in self.train_loader:
                features,targets = features.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                predictions = self(features)
                loss = criterion(predictions, targets)
                batch_train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            #epoch training loss
            train_loss = sum(batch_train_loss) / len(batch_train_loss)
            self.epoch_train_loss.append(train_loss)

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

                #flatten batch for input to mape function
                self.epoch_val_predictions = np.concatenate(batch_val_predictions, axis=0)
                self.epoch_val_actual = np.concatenate(batch_val_actual, axis=0) 

                # Calculate MAPE on the validation data
                val_mape = mean_absolute_percentage_error(self.epoch_val_actual, self.epoch_val_predictions)

                #epoch validation loss
                val_loss = sum(batch_val_loss) / len(batch_val_loss)
                self.epoch_val_loss.append(val_loss)

            #print(f"Epoch: {epoch+1} train loss: {self.epoch_train_loss[epoch]:.4f} val loss: {self.epoch_val_loss[epoch]:.4f} val MAPE: {val_mape:.4f}")
  
    #graph training and validation loss
    def graph(self):
        plt.figure(figsize=(12, 6))
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        epochs = range(1, self.num_epochs+1)
        plt.plot(epochs, self.epoch_train_loss, label="Training Loss")
        plt.plot(epochs, self.epoch_val_loss, label="Validation Loss")
        plt.legend()
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

        with torch.no_grad():
            self.most_recent_day = self.targets.tail(1).values
            self.most_recent_day = torch.tensor(self.most_recent_day, dtype=torch.float32).to(self.device)

            #forward pass on the most recent day
            for i in range(days):
                prediction = self(self.most_recent_day)
                predictions.append(prediction.cpu().numpy().flatten())

                #update input for the next day.
                self.most_recent_day = prediction

        return predictions
    

    def print_results(self, days):
        daily_prices = self.predict_ahead(days)

        for i in range(len(daily_prices)):
            print("Predicted Prices (Open  High  Low  Close):", 
            daily_prices[i][0], " | ", daily_prices[i][1], " | ", 
            daily_prices[i][2], " | ", daily_prices[i][3])