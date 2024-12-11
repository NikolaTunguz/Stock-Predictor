import sys
import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Models')))
from linear_regression_class import MyLinearRegression
from data import DataPreprocessing

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

def modelOutput(num_days):
    file_name = "../Models/data.csv"
    data_preprocessor = DataPreprocessing(file_name)
    dataset = data_preprocessor.preprocessing()

    features = dataset.drop(columns=["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close", "Volume", "Date"])
    labels = dataset[["tomorrow_open", "tomorrow_high", "tomorrow_low", "tomorrow_close"]]

    linear_regression_model = MyLinearRegression(features, labels)
    linear_regression_model.train()
    output = linear_regression_model.predict_ahead(num_days)

    high_predictions = []
    day_outputs = output.splitlines(' ')
    for temp_output in day_outputs:
        temp_output = temp_output.split()
        num = temp_output[8].strip(',')
        high_predictions.append(float(num))
        
    # Generate the plot and save it
    plot_data(dataset, num_days, high_predictions)

    return output, 'stock_prediction_plot.png'

def plot_data(dataset, num_days, high_predictions):
    last_5_days = dataset.tail(5)
    plt.figure(figsize=(10, 5))

    # Ensure the 'Date' column is in datetime format
    last_5_days.loc[:, 'Date'] = pd.to_datetime(last_5_days['Date'])

    plt.plot(last_5_days['Date'], last_5_days['High'], label='Actual High', marker='o')

    # Predict for the next 'num_days' and plot the predictions
    predictions_dates = pd.date_range(last_5_days['Date'].iloc[-1], periods = num_days + 1, freq='D')[1:]
    predictions_dates = pd.to_datetime(predictions_dates)  # Convert to datetime

    plt.plot(predictions_dates, high_predictions, label=f"Predicted High ({num_days} Days Ahead)", linestyle='--', marker='x')

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Stock Price Daily High')
    plt.title(f"High Stock Price for the Last 5 Days and {num_days} Days Prediction")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save the plot to display in the Gradio interface
    plt.savefig('stock_prediction_plot.png')
    plt.close()

demo = gr.Interface(
    fn=modelOutput,
    inputs=["number"],
    outputs=[gr.Textbox(), gr.Image(type="filepath")],
)

demo.launch()
