#overall imports
import data
import torch

#model imports
import linear_regression_class
import xgboost_class
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

    #declare models, passing in x and y data
    linear_regression_model = linear_regression_class.MyLinearRegression(features, labels)
    xgboost_model = xgboost_class.MyXGBoost(features, labels)

    nn_model = neural_network.NeuralNetwork(4, 4, features, labels)
    nn_model = nn_model.to(device)
    
    #train and evaluate models 
    linear_regression_model.train()
    linear_error = linear_regression_model.evaluate()

    xgboost_model.train()
    xgboost_error = xgboost_model.evaluate()

    print("Linear Test Dataset Percent Error: ", linear_error * 100)
    print("XGBoost Test Dataset Percent Error: ", xgboost_error * 100)

    nn_model.train_model()
    nn_model.print_results()

if __name__ == "__main__":
    main()
