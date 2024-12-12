#overall imports
import data
import torch

#model imports
import linear_regression_class
import xgboost_class
import neural_network
import lstm
import svr_class

import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #process/update data with new columns
    file_name = "data.csv"
    data_preprocessor = data.DataPreprocessing(file_name)
    dataset = data_preprocessor.preprocessing()

    #extracting x and y data
    features = dataset.drop(columns=["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close", "Volume", "Date"])
    labels = dataset[["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]]

    size = len(features)
    #features = features[int(size):]
    #labels = labels[int(size):]

    #declare models, passing in x and y data
    #linear_regression_model = linear_regression_class.MyLinearRegression(features, labels)
    #xgboost_model = xgboost_class.MyXGBoost(features, labels)
    #svr_model = svr_class.MySVR(features, labels)

    nn_model = neural_network.NeuralNetwork(4, 4, features, labels)
    nn_model = nn_model.to(device)

    lstm_model = lstm.LSTMPredictor(4, 32, 4, features, labels)
    lstm_model = lstm_model.to(device)
    
    #train and evaluate models 
    #linear_regression_model.train()
    #linear_error = linear_regression_model.evaluate()
    #linear_values = linear_regression_model.get_split_MAPE()

    #xgboost_model.train()
    #xgboost_error, xgboost_rmse = xgboost_model.evaluate()
    #xgboost_values = xgboost_model.get_split_MAPE()

    #svr_model.train()
    #svr_error, svr_rmse = svr_model.evaluate()
    #svr_values = svr_model.get_split_MAPE()


    #print("Linear Test Dataset Percent Error: ", linear_error * 100)
    #print("XGBoost Test Dataset Percent Error: ", xgboost_error * 100)
    #print("XGBoost Root Mean Squared Error: ", xgboost_rmse )
    #print("SVR Test Dataset Percent Error: ", svr_error * 100)
    #print("SVR Root Mean Squared Error: ", svr_rmse )

    
    nn_model.train_model()
    nn_model.print_results(1)
    print(nn_model.evaluate())

    lstm_model.train_model()
    lstm_model.print_results(1)
    print(lstm_model.evaluate())


    #plotting error of each model
    # for i in range(3):
    #     #print(linear_values[i])
    #     #print(xgboost_values[i])
    #     print(svr_values[i])
    #     pass

    #linear_regression_model.predict_ahead(5)
    #print()
    #xgboost_model.predict_ahead(5)
    # svr_model.predict_ahead(5)

    #xgboost_model.plot_metrics()

if __name__ == "__main__":
    main()
