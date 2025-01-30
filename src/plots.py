# plot.py
import numpy as np
import plotly.graph_objs as go
import pandas as pd
from src.utils import (
    load_poisson_model,
    load_transformer_model,
    input_dim,
    embed_dim,
    num_heads,
    num_layers,
    output_dim,
    max_length,
    sequence_length,
    TRANSFORMER_MODEL_BOWL_0_PATH,
    TRANSFORMER_MODEL_BOWL_1_PATH,
    POISSON_MODEL_PATH
)
from src.result_prediction import predict_with_transformer_model, predict_with_poisson_model

# Dictionary for price floor
price_floor_dict = {
    '0': 185.00, 'S': 159.00, '3': 149.00, '1': 147.00, '2': 138.00,
    'U': 132.00, '4': 123.00, '5': 115.00, '6': 100.00, 'T': 100.00,
    '7': 93.00, 'D': 92.00, '8': 86.00, 'H': 83.00, '9': 82.00, 'O': 79.00,
    'A': 76.00, 'E': 74.00, 'B': 68.00, 'F': 68.00, 'I': 65.00,
    'J': 65.00, 'C': 58.00, 'P': 55.00, 'G': 54.00, 'K': 51.00,
    'M': 47.00, 'L': 45.00, 'Q': 44.00, 'N': 38.00, 'R': 37.00
}

def get_transformer_demand_curve(model, preprocessor, filtered_X_test, opponent, price_code, ticket_prices):
    """
    Generates demand curve predictions using a transformer model for specified opponent and price code.
    Returns:
        List of predictions corresponding to each ticket price.
    """
    modified_entries = pd.DataFrame()
    for price in ticket_prices:
        temp_entries = filtered_X_test.copy()
        temp_entries.loc[(temp_entries['opponent'] == opponent) & (temp_entries['price_code'] == price_code), 'times_above_floor'] = price
        modified_entries = pd.concat([modified_entries, temp_entries], ignore_index=True)

    transformer_price_predictions = predict_with_transformer_model(model, preprocessor, modified_entries, sequence_length)
    predictions = []
    step = len(filtered_X_test[filtered_X_test['opponent'] == opponent])
    for i in range(0, len(transformer_price_predictions), step):
        predictions.append(transformer_price_predictions[i])

    return predictions

def get_poisson_demand_curve(model, filtered_X_test, opponent, price_code, ticket_prices):
    """
    Generates demand curve predictions using a Poisson model for specified opponent and price code.
    Returns:
        List of predictions corresponding to each ticket price.
    """
    predictions = []
    for price in ticket_prices:
        modified_entries = filtered_X_test.copy()
        modified_entries.loc[(modified_entries['opponent'] == opponent) & (modified_entries['price_code'] == price_code), 'times_above_floor'] = price
        poisson_predictions = predict_with_poisson_model(modified_entries, model)
        predictions.append(poisson_predictions[0])
    return predictions

def update_demand_curve_plot(filtered_X_test, preprocessor, opponent, price_code, model_type):
    """
    Updates the demand curve plot for a specified opponent and price code using either a Poisson or transformer model.
    Returns:
        Plotly Figure object representing the demand curve.
    """
    ticket_prices = np.linspace(1, 8, num=30)
    selected_entries = filtered_X_test[(filtered_X_test['opponent'] == opponent) & (filtered_X_test['price_code'] == price_code)]

    if selected_entries.empty:
        return go.Figure()

    bowl_location = selected_entries['bowl_location'].iloc[0]
    last_price = selected_entries['last_price'].iloc[0]
    price_floor = price_floor_dict[str(price_code)]
    calculate_date = selected_entries['calculate_date'].iloc[0]

    if model_type == 'poisson':
        poisson_model = load_poisson_model(POISSON_MODEL_PATH)
        predictions = get_poisson_demand_curve(poisson_model, filtered_X_test, opponent, price_code, ticket_prices)
    else:
        transformer_model_path = TRANSFORMER_MODEL_BOWL_0_PATH if bowl_location == 0 else TRANSFORMER_MODEL_BOWL_1_PATH
        transformer_model = load_transformer_model(transformer_model_path, input_dim, embed_dim, num_heads, num_layers, output_dim, max_length)
        predictions = get_transformer_demand_curve(transformer_model, preprocessor, filtered_X_test, opponent, price_code, ticket_prices)

    actual_prices = ticket_prices * price_floor
    actual_prices, predictions = zip(*sorted(zip(actual_prices, predictions)))

    demand_curve_fig = go.Figure()
    demand_curve_fig.add_trace(go.Scatter(x=actual_prices, y=predictions, mode='lines+markers', name='Predicted Tickets Sold'))
    demand_curve_fig.add_shape(
        type="line",
        x0=last_price,
        y0=min(predictions),
        x1=last_price,
        y1=max(predictions),
        line=dict(color="Red", dash="dash"),
    )
    demand_curve_fig.update_layout(
        title=f'Demand Curve for Opponent: {opponent}, Price Code: {price_code}, Bowl Location: {"Lower" if bowl_location == 0 else "Upper"} on {calculate_date}',
        xaxis_title='Ticket Price',
        yaxis_title='Predicted Tickets Sold',
        template='plotly_white'
    )

    return demand_curve_fig

def update_aggregate_sales_plot(filtered_X_test, preprocessor, opponent, model_type='poisson'):
    """
    Updates the aggregate sales plot for a specified opponent using a Poisson model.
    Returns:
        Plotly Figure object representing the aggregate sales.
    """
    times_above_floor_values = np.linspace(1, 8, num=30)
    selected_entries = filtered_X_test[filtered_X_test['opponent'] == opponent]

    if selected_entries.empty:
        return go.Figure()

    aggregate_predictions = []
    poisson_model = load_poisson_model(POISSON_MODEL_PATH)

    for taf_value in times_above_floor_values:
        modified_entries = filtered_X_test.copy()
        modified_entries['times_above_floor'] = taf_value
        poisson_predictions = predict_with_poisson_model(modified_entries, poisson_model)
        total_predicted_tickets = sum(poisson_predictions)
        aggregate_predictions.append(total_predicted_tickets)

    aggregate_sales_fig = go.Figure()
    aggregate_sales_fig.add_trace(go.Scatter(x=times_above_floor_values, y=aggregate_predictions, mode='lines+markers', name='Aggregate Sales'))
    aggregate_sales_fig.update_layout(
        title=f'Aggregate Sales for Opponent: {opponent}',
        xaxis_title='Times Above Floor',
        yaxis_title='Total Predicted Tickets Sold',
        template='plotly_white'
    )

    return aggregate_sales_fig

def update_price_change_impact_aggregate_plot(filtered_X_test, opponent):
    """
    Updates the aggregate demand plot to show the impact of changing ticket prices from 50% to 200% for a specified opponent using the Poisson model.
    
    Args:
        filtered_X_test: DataFrame containing test data.
        opponent: Opponent identifier.
        
    Returns:
        Plotly Figure object representing the impact of price changes on aggregate demand.
    """
    price_multipliers = np.linspace(0.5, 2.0, num=30)
    selected_entries = filtered_X_test[filtered_X_test['opponent'] == opponent]

    if selected_entries.empty:
        return go.Figure()

    calculate_date = selected_entries['calculate_date'].iloc[0]
    poisson_model = load_poisson_model(POISSON_MODEL_PATH)

    aggregate_predictions = []
    for multiplier in price_multipliers:
        modified_entries = selected_entries.copy()
        for price_code in price_floor_dict.keys():
            price_floor = price_floor_dict[price_code]
            condition = modified_entries['price_code'] == price_code
            modified_entries.loc[condition, 'times_above_floor'] = (
                modified_entries.loc[condition, 'times_above_floor'] * price_floor * multiplier / price_floor
            )
        poisson_predictions = predict_with_poisson_model(modified_entries, poisson_model)
        total_predicted_tickets = sum(poisson_predictions)
        aggregate_predictions.append(total_predicted_tickets)

    price_change_impact_fig = go.Figure()
    price_change_impact_fig.add_trace(go.Scatter(x=price_multipliers, y=aggregate_predictions, mode='lines+markers', name='Aggregate Predicted Tickets Sold'))
    price_change_impact_fig.add_shape(
        type="line",
        x0=1,
        y0=min(aggregate_predictions),
        x1=1,
        y1=max(aggregate_predictions),
        line=dict(color="Red", dash="dash"),
    )
    price_change_impact_fig.update_layout(
        title=f'Impact of Price Changes on Aggregate Sales for Opponent: {opponent}',
        xaxis_title='Change Price By (For all Price Code)',
        yaxis_title='Total Predicted Tickets Sold',
        template='plotly_white'
    )

    return price_change_impact_fig
