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

    #call models
    linear_regression = None
    neural_network = None

    

if __name__ == "__main__":
    main()
