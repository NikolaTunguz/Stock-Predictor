#nn imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#data imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import data 
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    #define global variables and model
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_train_loss = []
        self.epoch_val_loss = []
        self.epoch_val_predictions = []
        self.epoch_val_actual = []
        self.num_epochs = None

        hidden_size1 = 256
        hidden_size2 = 128

        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size2, output_dim)

    #forward pass through model
    def forward(self, x):
        output = self.fc1(x)
        output = self.activation1(output)

        output = self.fc2(output)
        output = self.activation2(output)
        output = self.drop(output)

        output = self.fc3(output)
        return output
        
    def prepare_data(self, file):
        #set up generic CSV
        data_processing = data.DataPreprocessing(file)
        dataset = data_processing.preprocessing()

        #drop last row for training
        self.most_recent_day = dataset[["Open", "High", "Low", "Close"]].tail(1).values
        self.most_recent_day = torch.tensor(self.most_recent_day, dtype=torch.float32)
        dataset.drop(dataset.tail(1).index, inplace=True)

        #define features and targets
        features = dataset[["Open", "High", "Low", "Close"]].values #values on a day
        targets = dataset[["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]].values #"next day" values

        #split dataset
        #not shuffled so that the data maintains chronological order
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(features, targets, test_size=0.2, random_state=42, shuffle=False) 

        #convert to tensors
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.x_val = torch.tensor(self.x_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)

        #place datasets in dataloader
        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        batch_size = 64
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train_model(self, optimizer, criterion, num_epochs):
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

            print(f"Epoch: {epoch+1} train loss: {self.epoch_train_loss[epoch]:.4f} val loss: {self.epoch_val_loss[epoch]:.4f} val MAPE: {val_mape:.4f}")
  

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

    #next day prediction
    def predict(self):
        self.eval()
        with torch.no_grad():
            self.most_recent_day = self.most_recent_day.to(self.device)

            #forward pass on the most recent day
            prediction = self(self.most_recent_day)
            return prediction.cpu().numpy().flatten()
        
def main():
    #4 input features 4 target features
    model = NeuralNetwork(4, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #hyperparameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 20

    #train model
    file = "data.csv"
    model.prepare_data(file)
    model.train_model(optimizer, criterion, num_epochs)

    #training graph
    model.graph()

    #prediction based on last day
    tomorrow_prices = model.predict()
    
    # Print the predicted prices
    print("Tommorow's Predicted Prices (Open  High  Low  Close):", tomorrow_prices[0], " | ", tomorrow_prices[1], " | ", tomorrow_prices[2], " | ", tomorrow_prices[3])

if __name__ == "__main__":
    main()

        



        



  