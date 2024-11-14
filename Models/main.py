#overall imports
import data

#model imports
import linear_regression
import neural_network

def main():
    #process/update data with new columns
    file_name = "data.csv"
    data_preprocessor = data.DataPreprocessor(file_name)
    dataset = data_preprocessor.preprocessing()

    #extracting x and y data
    labels = dataset["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]
    features = dataset.drop(columns = ["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"])

    #declare models, passing in x and y data
    linear_regression = linear_regression.LinearRegression(features, labels)
    neural_network = None

    #train and evaluate models 
    linear_regression.train()
    linear_accuracy = linear_regression.evaluate()

    print(linear_accuracy)

if __name__ == "__main__":
    main()
