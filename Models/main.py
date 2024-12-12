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

    #size = len(features)
    #features = features[int(size):]
    #labels = labels[int(size):]

    #declare models, passing in x and y data
    linear_regression_model = linear_regression_class.MyLinearRegression(features, labels)
    xgboost_model = xgboost_class.MyXGBoost(features, labels)
    svr_model = svr_class.MySVR(features, labels)
    nn_model = neural_network.NeuralNetwork(4, 4, features, labels)
    lstm_model = lstm.LSTMPredictor(4, 32, 4, features, labels)

    #putting nn models to device
    nn_model = nn_model.to(device)
    lstm_model = lstm_model.to(device)
    
    #train and evaluate models 
    linear_regression_model.train()
    linear_mape, linear_rmse = linear_regression_model.evaluate()
    linear_split_mape = linear_regression_model.get_split_MAPE()
    linear_split_rmse = linear_regression_model.get_split_RMSE()

    xgboost_model.train()
    xgboost_mape, xgboost_rmse = xgboost_model.evaluate()
    xgboost_split_mape = xgboost_model.get_split_MAPE()
    xgboost_split_rmse = xgboost_model.get_split_RMSE()

    svr_model.train()
    svr_mape, svr_rmse = svr_model.evaluate()
    svr_split_mape = svr_model.get_split_MAPE()
    svr_split_rmse = svr_model.get_split_RMSE()

    nn_model.train_model()
    nn_mape, nn_rmse = nn_model.evaluate()
    nn_split_mape = nn_model.get_split_MAPE()
    nn_split_rmse = nn_model.get_split_RMSE()

    lstm_model.train_model()
    lstm_mape, lstm_rmse = lstm_model.evaluate()
    lstm_split_mape = lstm_model.get_split_MAPE()
    lstm_split_rmse = lstm_model.get_split_RMSE()


    #outputting test mape and rmse
    print("Linear Test Dataset Percent Error: ", linear_mape * 100, "RMSE: ", linear_rmse)
    print("XGBoost Test Dataset Percent Error: ", xgboost_mape * 100, "RMSE: ", xgboost_rmse)
    print("SVR Test Dataset Percent Error: ", svr_mape * 100, "RMSE: ", svr_rmse)
    print("NN Test Dataset Percent Error: ", nn_mape * 100, "RMSE: ", nn_rmse)
    print("LSTM Test Dataset Percent Error: ", lstm_mape * 100, "RMSE: ", lstm_rmse)

    #outputting 3 day prediction of each model for comparison
    print("3 Day Predictions: ")

    print("Linear Regression")
    print(linear_regression_model.predict_ahead(3))

    print("XGBoost")
    print(xgboost_model.predict_ahead(3))

    print("Support Vector Regression")
    print(svr_model.predict_ahead(3))

    print("Neural Network")
    print(nn_model.predict_ahead(3))

    print("Long Short-Term Memory")
    print(lstm_model.predict_ahead(3))

    #graphing nn figures
    nn_model.graph()
    lstm_model.graph()

    #plotting mape error of each model on train/val/test splits
    mape_y = []
    for i in range(3):
        temp = []
        temp.append(linear_split_mape[i])
        temp.append(xgboost_split_mape[i])
        temp.append(svr_split_mape[i])
        temp.append(nn_split_mape[i])
        temp.append(lstm_split_mape[i])
        mape_y.append(temp)

    #plotting rmse error of each model
    rmse_y = []
    for i in range(3):
        temp = []
        temp.append(linear_split_rmse[i])
        temp.append(xgboost_split_rmse[i])
        temp.append(svr_split_rmse[i])
        temp.append(nn_split_rmse[i])
        temp.append(lstm_split_rmse[i])
        rmse_y.append(temp)

    models_x = ["Linear Regression", "XGBoost", "Support Vector Regression", "Neural Network", "Long Short-Term Memory"]

    #actually plotting
    plt.figure(figsize = (15, 10))

    #MAPE
    plt.subplot(1,2,1)
    plt.title("MAPE over Models")
    plt.xlabel("Models")
    plt.ylabel("Mean Absolute Percent Error")
    plt.plot(models_x, mape_y[0], label = "Train MAPE", c = "orange", marker = "o", linestyle = "--")
    plt.plot(models_x, mape_y[1], label = "Validation MAPE", c = "teal", marker = "v", linestyle = "--")
    plt.plot(models_x, mape_y[2], label = "Test MAPE", c = "black", marker = "x", linestyle = "--")
    plt.xticks(rotation=45)
    plt.legend()

    #RMSE
    plt.subplot(1,2,2)
    plt.title("RMSE over Models")
    plt.xlabel("Models")
    plt.ylabel("Root Mean Squared Error")
    plt.plot(models_x, rmse_y[0], label = "Train RMSE", c = "orange", marker = "o", linestyle = "--")
    plt.plot(models_x, rmse_y[1], label = "Validation RMSE", c = "teal", marker = "v", linestyle = "--")
    plt.plot(models_x, rmse_y[2], label = "Test RMSE", c = "black", marker = "x", linestyle = "--")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
