import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

alt.data_transformers.enable('vegafusion')

def plot_residuals(poi_model, y_train, X_train):
    """
    Calculates residuals and outliers, then generates a scatterplot of 
    the residuals. 

    Parameters
    ----------
    poi_model : sklearn.linear_model.PoissonRegressor
        Poisson regression model loaded from joblib file.

    y_train : pandas.DataFrame
        Dataframe containing the actual target values.

    X_train : pandas.DataFrame 
        File path to the X_train parquet file containing features.

    Returns
    -------
    list of altair.Chart objects
        List containing Altair chart objects for residual scatter plots.
    """
    # Predict & calculate residuals 
    y_train_values = y_train['target_host_sold-today'].values
    y_pred = poi_model.predict(X_train)
    X_train['residuals'] = y_train_values - y_pred

    # Calculate extreme outliers 
    Q1 = X_train['residuals'].quantile(0.05)
    Q3 = X_train['residuals'].quantile(0.95)
    IQR = Q3 - Q1
    lower_threshold = Q1 - 1.5 * IQR
    upper_threshold = Q3 + 1.5 * IQR

    # Mark outliers in the data from Bieksa's event 
    X_train['is_outlier'] = ((X_train['residuals'] < lower_threshold) | (X_train['residuals'] > upper_threshold )) & (X_train['event_id'] == '11005D05E8D65350')

    # Create plot objects for example features 
    features = ['host_sold-total', 'opponent']
    labels = ['Total Host Tickets Sold', 'Opponent']
    plots = []

    for i, feature in enumerate(features):
        scatter = alt.Chart(X_train).mark_circle().encode(
            x=alt.X(feature, title=f'{labels[i]}'),
            y=alt.Y('residuals:Q', title='Residual'), 
            color=alt.condition(
                alt.datum.is_outlier,
                alt.value('rgba(255, 0, 0, 0.7)'),  
                alt.value('rgba(70, 130, 180, 0.7)')
            )
        ).properties(
            title=f"Residual Outliers in `{feature}` Feature",
            width=450
        ).configure_axis(
            titleFontSize=16, 
            labelFontSize=14
        ).configure_title(
            fontSize=18 
        )
        plots.append(scatter)

    host_plot = plots[0]
    opponent_plot = plots[1].configure_axisX(labelAngle=45).properties(width=600)

    return [host_plot, opponent_plot]

def plot_coefficients(coef_df):
    """
    Generates a bar chart of the top 20 coefficents with the highest magnitude from
    the poisson model.

    Parameters
    ----------
    coef_df : pandas.DataFrame
        Dataframe containing coefficients and feature names.

    Returns
    -------
    altair.Chart object
        Altair chart object for coefficient bar chart.
    """
    # Filter for top 20 
    coef_df = coef_df.head(20)

    # Create plot 
    coef_bar = alt.Chart(coef_df).mark_bar().encode(
        x=alt.X('coefficients', title='Coefficient', scale=alt.Scale(domain=(-0.25, 0.25))),
        y=alt.Y('feature', title='Feature', sort=None),
        color=alt.condition(
            alt.datum.coefficients > 0,
            alt.value('steelblue'), 
            alt.value('orange')   
        )
    ).properties(
        width=600,
        height=400,
        title=alt.TitleParams('Poisson Top 20 features with Highest Magnitude', fontSize=20, anchor='middle'),
    ).configure_axis(
        titleFontSize=16, 
        labelFontSize=14
    )

    return coef_bar

def save_plots(plot, plot_directory, plot_name):
    """
    Saves generated plots as a png to specified paths.

    Parameters
    ----------
    plot : altair.Chart object
        Plot object to save. 

    plot_path : str
        Path to the directory where plots will be saved.

    plot_name : str
        File name for the plot. 
    
    Returns
    -------
    None   
    """
    plot_path = os.path.join(plot_directory, plot_name)
    plot.save(plot_path)

def correlated_feat_map(df, plot_directory, plot_name):
    """
    Generates a heatmap for highly correlated features in df. Saves the
    plot as a png in the specified directory. 

    Parameters
    ----------
    df: pandas.Dataframe 
        Dataframe containing columns to plot. 

    plot_directory : str
        Path to the directory where plot will be saved.

    plot_name : str
        File name for the plot.

    Returns
    -------
    None
    """
    # Select numeric columns 
    df_numeric= df.select_dtypes(include=[float, int])
    df_numeric.drop(columns=['platinum_sold_-_total', 'patinum_asp_-_total'], inplace=True)

    # Calculate correlations
    corr_matrix = df_numeric.corr()

    # Filter for highly correlated features
    melted_corr_matrix = corr_matrix.reset_index().melt(id_vars='index', var_name='feature_2', value_name='correlation')
    melted_corr_matrix.columns = ['feature_1', 'feature_2', 'correlation']
    high_corr = melted_corr_matrix[
        (melted_corr_matrix['correlation'].abs() >= 0.8) & 
        (melted_corr_matrix['feature_1'] != melted_corr_matrix['feature_2'])
    ]
    high_corr_features = pd.unique(high_corr[['feature_1', 'feature_2']].values.ravel('K'))
    filtered_corr_matrix = corr_matrix.loc[high_corr_features, high_corr_features]

    # Create plot 
    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True,
                linewidths=0.5, linecolor='white', cbar_kws={"shrink": 0.75})
    plt.title('Highly Correlated Features', fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Feature', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save image 
    plot_path = plot_directory + '/' + plot_name
    plt.savefig(plot_path, bbox_inches='tight') 

def main(): 
    """
    Main function to orchestrate functions to generate and save the final report plots. 
    """
    # Location to save plots
    plot_directory = 'output/img/'

    # Load required data and objects  
    poi_model = joblib.load('output/model/poisson_model.joblib')
    y_train = pd.read_parquet('data/output/y_train.parquet', engine='pyarrow')
    X_train = pd.read_parquet('data/output/X_train.parquet', engine='pyarrow')
    coef_df = pd.read_csv('data/output/poisson_feature_importances.csv')
    feat_eng_df = pd.read_parquet('data/output/feature_engineered.parquet', engine='pyarrow')

    # Generate residual plots
    residual_plots = plot_residuals(poi_model, y_train, X_train)

    # Generate coefficient bar chart
    coef_bar = plot_coefficients(coef_df)

    # Save plots to file
    save_plots(residual_plots[0], plot_directory, 'residuals_host_sold-total.png')
    save_plots(residual_plots[1], plot_directory, 'residuals_opponent.png')
    save_plots(coef_bar, plot_directory,'poisson_coefficients.png')

    # Generate and save heat map 
    correlated_feat_map(feat_eng_df, plot_directory, 'feature_correlation_map.png')

if __name__ == "__main__":
    main()
