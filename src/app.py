# app.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import glob
from dash import dash_table
import plotly.express as px
from src.plots import update_demand_curve_plot, update_aggregate_sales_plot, update_price_change_impact_aggregate_plot
from src.preprocess_feature_engineering import preprocess_data, feature_engineer_data
from src.utils import preprocess_train_data, START_DATE, DATA_PATH, PREDICTION_INPUT_PATH, OUTPUT_FILENAME_TEMPLATE

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# Load and preprocess data
X_train = pd.read_parquet(DATA_PATH)

# Read all CSV files from data/prediction_input and concatenate them
input_files = glob.glob(PREDICTION_INPUT_PATH)
all_data = pd.concat([pd.read_csv(file) for file in input_files])

# Preprocess and feature engineer data
all_data = preprocess_data(all_data, START_DATE)
all_data = feature_engineer_data(all_data)

# Filter the last calculate_date
last_calculate_date = all_data['calculate_date'].max().strftime('%Y-%m-%d')
filtered_X_test = all_data[all_data['calculate_date'] == last_calculate_date]

# Preprocess data
preprocessor = preprocess_train_data(X_train)

# Load predictions
predictions_df = pd.read_csv(OUTPUT_FILENAME_TEMPLATE.format(date=last_calculate_date))

# Calculate highest predicted sales event
highest_event = predictions_df.groupby('event_name')['predicted_tickets_sold_transformer'].sum().idxmax()
highest_event_value = predictions_df.groupby('event_name')['predicted_tickets_sold_transformer'].sum().max()

# Calculate highest combination of predicted sale based on 'price_code', 'opponent'
highest_combination = predictions_df.loc[predictions_df['predicted_tickets_sold_transformer'].idxmax()]
highest_combination_value = highest_combination['predicted_tickets_sold_transformer']
highest_combination_text = f"Price Code: {highest_combination['price_code']}, Opponent: {highest_combination['opponent']}"

# Function to generate the layout for the demand curve tab
def render_demand_curve_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Select Opponent'),
                dcc.Dropdown(
                    id='opponent-dropdown',
                    options=[{'label': opponent, 'value': opponent} for opponent in filtered_X_test['opponent'].unique()],
                    value=filtered_X_test['opponent'].unique()[0]
                )
            ], width=4),
            dbc.Col([
                html.Label('Select Price Code'),
                dcc.Dropdown(
                    id='price-code-dropdown',
                    options=[{'label': code, 'value': code} for code in filtered_X_test['price_code'].unique()],
                    value=filtered_X_test['price_code'].unique()[0]
                )
            ], width=4),
            dbc.Col([
                html.Label('Select Model'),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Poisson Model', 'value': 'poisson'},
                        {'label': 'Transformer Model', 'value': 'transformer'}
                    ],
                    value='poisson'
                )
            ], width=4)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='demand-curve-plot')
            ], width=12)
        ])
    ])

# Function to generate the layout for the combined sales tab
def render_combined_sales_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Select Opponent'),
                dcc.Dropdown(
                    id='opponent-dropdown-2',
                    options=[{'label': opponent, 'value': opponent} for opponent in filtered_X_test['opponent'].unique()],
                    value=filtered_X_test['opponent'].unique()[0]
                )
            ], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='aggregate-sales-plot')
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='price-change-impact-plot')
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='bar-plot')
            ], width=12)
        ])
    ])

# Function to generate the layout for the predictions table tab
def render_predictions_table_tab():
    top_20_predictions = predictions_df.nlargest(20, 'predicted_tickets_sold_transformer')
    return html.Div([
        dash_table.DataTable(
            id='predictions-table',
            columns=[
                {'name': 'Event Name', 'id': 'event_name'},
                {'name': 'Event Date', 'id': 'event_date'},
                {'name': 'Opponent', 'id': 'opponent'},
                {'name': 'Price Code', 'id': 'price_code'},
                {'name': 'Last Price', 'id': 'last_price'},
                {'name': 'Host Sold Yesterday', 'id': 'host_sold-yesterday'},
                {'name': 'Predicted Tickets Sold (Transformer)', 'id': 'predicted_tickets_sold_transformer'},
                {'name': 'Predicted Tickets Sold (Poisson)', 'id': 'predicted_tickets_sold_poisson'},
            ],
            data=top_20_predictions.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            sort_action='native'
        )
    ])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(f"Dashboard of Predicted Demand Curves on {last_calculate_date}", className="text-center m-1")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Highest Predicted Sales Event"),
                dbc.CardBody([
                    html.H3(highest_event, className="card-title"),
                    html.H5(f"Value: {highest_event_value:.2f}", className="card-text")
                ])
            ], color="primary", inverse=True, className="m-2")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Highest Combination Predicted Sale"),
                dbc.CardBody([
                    html.H3(highest_combination_text, className="card-title"),
                    html.H5(f"Value: {highest_combination_value:.2f}", className="card-text")
                ])
            ], color="info", inverse=True, className="m-2")
        ], width=6)
    ], className="m-2"),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Demand Curve', value='tab-1'),
        dcc.Tab(label='Combined Sales (Poisson)', value='tab-2'),
        dcc.Tab(label='Predictions Table', value='tab-3')
    ], className="m-2"),
    html.Div(id='tabs-content', className="m-2")
], fluid=True)

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-1':
        return render_demand_curve_tab()
    elif tab == 'tab-2':
        return render_combined_sales_tab()
    elif tab == 'tab-3':
        return render_predictions_table_tab()

@app.callback(
    Output('demand-curve-plot', 'figure'),
    [Input('opponent-dropdown', 'value'), Input('price-code-dropdown', 'value'), Input('model-dropdown', 'value')]
)
def update_demand_curve(opponent, price_code, model_type):
    filtered_data = filtered_X_test[(filtered_X_test['opponent'] == opponent) & (filtered_X_test['price_code'] == price_code)]
    return update_demand_curve_plot(filtered_data, preprocessor, opponent, price_code, model_type)

@app.callback(
    Output('aggregate-sales-plot', 'figure'),
    Input('opponent-dropdown-2', 'value')
)
def update_aggregate_sales(opponent):
    filtered_data = filtered_X_test[filtered_X_test['opponent'] == opponent]
    return update_aggregate_sales_plot(filtered_data, preprocessor, opponent, model_type='poisson')

@app.callback(
    Output('bar-plot', 'figure'),
    Input('opponent-dropdown-2', 'value')
)
def update_bar_plot(opponent):
    opponent_sales = predictions_df.groupby('opponent')['predicted_tickets_sold_poisson'].sum().reset_index()
    bar_fig = px.bar(opponent_sales, x='opponent', y='predicted_tickets_sold_poisson', 
                     color='opponent', title='Total Predicted Tickets Sold by Opponent (Poisson Model)',
                     labels={'predicted_tickets_sold_poisson': 'Total Predicted Tickets Sold', 'opponent': 'Opponent'})
    bar_fig.update_layout(template='plotly_white')
    return bar_fig

@app.callback(
    Output('price-change-impact-plot', 'figure'),
    Input('opponent-dropdown-2', 'value')
)
def update_price_change_impact(opponent):
    filtered_data = filtered_X_test[filtered_X_test['opponent'] == opponent]
    return update_price_change_impact_aggregate_plot(filtered_data, opponent)

if __name__ == '__main__':
    app.run_server(debug=True)
