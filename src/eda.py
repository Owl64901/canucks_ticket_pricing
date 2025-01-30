import pandas as pd
import plotly.express as px
import altair as alt
import plotly.io as pio
#alt.data_transformers.enable('vegafusion')
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", message="I don't know how to infer vegalite type from 'empty'.  Defaulting to nominal.")

def main():
    df = pd.read_parquet('data/output/processed.parquet')

    # change of last price for one game
    df_filtered = df[(df['event_id'] == '11005B3CECED2D64')]
    df_filtered = df_filtered[df_filtered['price_code'].isin(['9', 'B', 'E', '4'])]
    legend_selection = alt.selection_point(fields=['price_code'], bind='legend')
    chart = alt.Chart(df_filtered).mark_circle(size=60).encode(
        x='calculate_date:T',
        y='last_price:Q',
        color='price_code:N',
        tooltip=['calculate_date', 'last_price', 'price_code'],
        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0))
    ).properties(
        width=600,
        height=400,
        title='Time Series of Last Price'
    ).add_params(
        legend_selection
    )
    chart.save('output/eda_img/price_change.png')

    # time series of number of ticket sold for one game
    line = alt.Chart(df_filtered).mark_line(point=True).encode(
        x=alt.X('calculate_date:T', title='Calculate Date'),
        y=alt.Y('ticket_sold-total:Q', title='Tickets Sold'),
        color=alt.Color('price_code:N', title='Price Code', legend=alt.Legend(title="Select Price Level")),
        tooltip=[alt.Tooltip('calculate_date:T', title='Calculate Date'), 
                 alt.Tooltip('ticket_sold-total:Q', title='Tickets Sold'),
                 alt.Tooltip('price_code:N', title='Price Code')],
        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0))
    ).add_params(
        legend_selection
    ).properties(
        title='Ticket Sales Distribution Across Price Levels',
        width=800,
        height=400
    )
    line.save('output/eda_img/price_onegame.png')

    # box plot of price level for all games 
    df['median'] = df.groupby('price_code')['last_price'].transform('median')
    df = df.sort_values('median')

    fig = px.box(df, x="price_code", y="last_price", boxmode='overlay')
    fig.update_traces(boxpoints=False)
    fig.update_layout(
        title="Boxplot of Last Price Grouped by Price Code",
        xaxis_title="Price Code",
        yaxis_title="Last Price"
    )
    pio.write_image(fig, 'output/eda_img/price_box.png')

    # average price over 3 seasons
    grouped_df = df.groupby(['price_code', 'event_date'])['initial_price'].mean().reset_index()
    grouped_df['event_date'] = pd.to_datetime(grouped_df['event_date'])
    filtered_df = grouped_df[grouped_df['event_date'] >= pd.Timestamp('2021-07-01')]
    filtered_df = filtered_df[filtered_df['price_code'].isin(['0', 'L'])]
    def map_event_date(date):
        if date < pd.Timestamp('2022-07-01'):
            return "2021-2022"
        elif date < pd.Timestamp('2023-07-01'):
            return "2022-2023"
        else:
            return "2023-2024"
    filtered_df['event_date'] = pd.to_datetime(filtered_df['event_date'])
    filtered_df['Season'] = filtered_df['event_date'].apply(map_event_date)
    filtered_df = filtered_df[filtered_df['initial_price'] <= 650]
    selection = alt.selection_point(fields=['price_code'], bind='legend')
    chart = alt.Chart(filtered_df).mark_line(strokeDash=[6, 2], point=True).encode(
        x=alt.X('event_date:T', title='Event Date', scale=alt.Scale()), 
        y=alt.Y('initial_price:Q', title='Average Ticket Price', scale=alt.Scale(zero=False, padding=5, domain=[0, 620])),  
        color=alt.Color('price_code:N', legend=alt.Legend(title="Area Code")),
        tooltip=[alt.Tooltip('event_date:T', title='Event Date'), 
                 alt.Tooltip('initial_price:Q', title='Initial Price'),
                 alt.Tooltip('price_code:N', title='Price Code')],
        opacity=alt.condition(selection, alt.value(1), alt.value(0))
    ).add_params(
        selection
    ).properties(
        width=500,
        height=400
    ).facet(
        column=alt.Column('Season:N', title='',  header=alt.Header(labelFontSize=20))
    ).resolve_scale(
        x='independent'  
    )
    chart = chart.configure_title(fontSize=15)
    chart = chart.configure_axis(labelFontSize=13, titleFontSize=15)
    chart.save('output/eda_img/avgprice.png')

    #histogram of counts of number of tickets sold
    sold_df = df[(df['opens'] > 0) & (df['last_price'] > 0)]
    plot_df = sold_df.melt(id_vars='price_code', value_vars=['host_sold-last_7days', 'host_sold-yesterday'], 
                    var_name='Time Period', value_name='Number of Tickets Sold')
    fig = px.histogram(plot_df, x="Number of Tickets Sold", color="Time Period", 
                    facet_col="Time Period", range_x=[-1, 50], range_y=[0,400000])
    fig.update_layout(showlegend=False, title_text="Number of Tickets Sold",
                    yaxis_title="Count", bargap=0.1)
    pio.write_image(fig, 'output/eda_img/count_ticket_sold.png')

if __name__ == "__main__":
    main()

