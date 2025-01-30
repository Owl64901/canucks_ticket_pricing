import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def plot_demand_curve(model_path, data_path, indices):
    # Load the model
    model = joblib.load(model_path)

    # Load the data
    X_train = pd.read_parquet(data_path)

    # Plotting setup
    plt.figure(figsize=(10, 8))

    # Iterate over each sample entry index
    for index in indices:
        # Select a sample entry
        sample_entry = X_train.iloc[index:index+1]

        # Extract price_code of the sample to find min and max prices in the same area
        price_code = sample_entry['price_code'].iloc[0]
        relevant_prices = X_train[X_train['price_code'] == price_code]['last_price']

        # Define a range of ticket prices to test based on the min and max of the relevant prices
        min_price = relevant_prices.min()
        max_price = relevant_prices.max()
        ticket_prices = np.linspace(min_price, max_price, num=30)

        # Store predictions
        predictions = []

        # Change the ticket price in the sample entry and predict tickets sold
        for price in ticket_prices:
            modified_entry = sample_entry.copy()
            modified_entry['last_price'] = price
            predicted_tickets_sold = model.predict(modified_entry)[0]
            predictions.append(predicted_tickets_sold)

        # Plotting the demand curve for the current sample
        plt.plot(ticket_prices, predictions, marker='o', label=f'Entry at Index {index}')

    # Final plot adjustments
    plt.title('Predicted Tickets Sold vs. Ticket Price')
    plt.xlabel('Ticket Price')
    plt.ylabel('Predicted Tickets Sold')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()

if __name__ == "__main__":
    model_path = 'output/model/baseline_linear_model.joblib'
    data_path = 'data/output/X_train.parquet'
    indices = [10000, 30000, 50000]
    plot_demand_curve(model_path, data_path, indices)