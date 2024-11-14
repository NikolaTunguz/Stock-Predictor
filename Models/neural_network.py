import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd
import data 

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.epoch_train_loss = []
        self.epoch_train_acc = []
        self.epoch_val_loss = []
        self.epoch_val_acc = []
        self.num_epochs = None


        hidden_size1 = 256
        hidden_size2 = 128

        self.fc1 = nn.Linear(input_dim, hidden_size1)
        self.activation1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_dim)

    def forward(self, x):
        output = self.fc1(x)
        output = self.activation1(output)
        output = self.fc2(output)
        output = self.activation2(output)
        output = self.fc3(output)
        return output
        
    def prepare_data(self, file):
        #set up generic CSV
        dataset = data.DataPreprocessing(file)
        dataset = dataset.read_data()
        dataset = dataset.preprocessing()

        #define features and targets
        features = dataset["Open", "High", "Low", "Close"]
        targets = dataset["tommorow_open", "tommorow_high", "tommorow_low", "tommorow_close"]

        #split dataset
        self.x_train, self.y_val, self.y_train, self.y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
        self.x_train = torch.tensor(self.x_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.x_train, dtype=torch.long)
        self.x_val = torch.tensor(self.x_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.long)
        
        #normalize data
        normalize = StandardScaler()
        self.x_train = normalize.fit_transform(self.x_train)
        self.x_test = normalize.transform(self.x_test)

        #place datasets in dataloader
        train_dataset = TensorDataset(self.x_train, self.y_train)
        val_dataset = TensorDataset(self.x_val, self.y_val)
        batch_size = 64
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train_model(self, optimizer, criterion, num_epochs):
        for epoch in range(num_epochs):
            self.train()
            batch_train_loss = []
            batch_train_acc = []
            batch_val_loss = []
            batch_val_acc = []

            #training
            for(features, targets) in self.train_loader:
                features,targets = features.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                predictions = self(features)

                loss = criterion(predictions, targets)
                batch_train_loss.append(loss.item())

                loss.backward()
                optimizer.step()

                #batch accuracy
                _, prediction_labels = torch.max(predictions, dim=1)
                labels = labels.cpu().numpy()
                prediction_labels = prediction_labels.cpu().numpy()
                batch_train_acc.append(accuracy_score(prediction_labels, labels))

            #epoch accuracy and loss
            train_acc = sum(batch_train_acc) / len(batch_train_acc)
            self.epoch_train_acc.append(train_acc)
            train_loss = sum(batch_train_loss) / len(batch_train_loss)
            self.epoch_val_loss.append(train_loss)

            #validation
            self.eval()
            with torch.no_grad():
                for(features,targets) in self.val_loader:
                    features,targets = features.to(self.device), targets.to(self.device)

                    predictions = self(data)

                    loss = criterion(predictions, targets)
                    batch_val_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    #batch accuracy
                    _, prediction_labels = torch.max(predictions, dim=1)
                    labels = labels.cpu().numpy()
                    prediction_labels = prediction_labels.cpu().numpy()
                    batch_val_acc.append(accuracy_score(prediction_labels, labels))
            
            #epoch accuracy and loss
            val_acc = sum(batch_val_acc) / len(batch_val_acc)
            self.epoch_val_acc.append(val_acc)
            val_loss = sum(batch_val_loss) / len(batch_val_loss)
            self.epoch_val_loss.append(val_loss)

        print(f"Epoch: {epoch+1} train acc: {self.epoch_train_acc[epoch]:.4f} train loss: {self.epoch_train_loss[epoch]:.4f} val acc: {self.epoch_val_acc[epoch]:.4f} val loss: {self.epoch_val_loss[epoch]:.4f}")

def main():
    #4 input features 4 target features
    model = NeuralNetwork(4, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5

    #train model
    model.train_model(optimizer, criterion, num_epochs)


if __name__ == "__main__":
    main()

        



        



