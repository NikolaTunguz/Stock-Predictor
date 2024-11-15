#overall imports
import data

#model imports
import linear_regression_class
import xgboost_class
import neural_network


def main():
    #process/update data with new columns
    file_name = "data.csv"
    data_preprocessor = data.DataPreprocessing(file_name)
    dataset = data_preprocessor.preprocessing()

    #extracting x and y data
    labels = dataset[["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]]
    features = dataset.drop(columns = ["Date", "tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"])

    #declare models, passing in x and y data
    linear_regression_model = linear_regression_class.MyLinearRegression(features, labels)
    xgboost_model = xgboost_class.MyXGBoost(features, labels)

    neural_network = None

    #train and evaluate models 
    linear_regression_model.train()
    linear_error = linear_regression_model.evaluate()

    xgboost_model.train()
    xgboost_error = xgboost_model.evaluate()



    print("Linear Test Dataset Percent Error: ", linear_error * 100)
    print("XGBoost Test Dataset Percent Error: ", xgboost_error * 100)

if __name__ == "__main__":
    main()
