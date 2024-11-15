#overall imports
import data
import torch

#model imports
import linear_regression
import neural_network

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #process/update data with new columns
    file_name = "data.csv"
    data_preprocessor = data.DataPreprocessing(file_name)
    dataset = data_preprocessor.preprocessing()

    #extracting x and y data
    features = dataset.drop(columns=["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close", "Volume", "Date"])
    labels = dataset[["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]]

    #declare models
    lr_model = linear_regression.LinearRegression(features, labels)

    nn_model = neural_network.NeuralNetwork(4, 4, features, labels)
    nn_model = nn_model.to(device)
    
    #train and evaluate models 
    lr_model.train()
    linear_accuracy = lr_model.evaluate()
    print(linear_accuracy)

    nn_model.train_model()
    nn_model.print_results()



if __name__ == "__main__":
    main()
