'''
 # @ Create Time: 2023-11-26 14:46:34.011634
'''
# Importing the necessary files
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.io as pio
from dash import Dash, dcc, html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__, title="MyVisual")

# Declare server for Heroku deployment. Needed for Procfile.
server = app.server

# Data loading and preprocessing
stock_data = pd.read_csv("data.csv")
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'].str.replace(',', ''), errors='coerce')

# Convert 'Date' column to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

current_time_str = stock_data['Date'].max()
print(current_time_str)
current_time = current_time_str.to_pydatetime()
print(current_time)

# Create a dropdown filter for company selection
companies = stock_data['Name'].unique().tolist()

# Calculate VWAP for each company
stock_data['VWAP'] = (stock_data['Closing_Price'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()

# Calculate Simple Moving Average (SMA) for Closing_Price (taking a window of 3 days as an example)
stock_data['SMA'] = stock_data['Closing_Price'].rolling(window=3).mean()

# Calculate Bollinger Bands
window = 20  # You can adjust the window size as needed
stock_data['Middle_Band'] = stock_data['Closing_Price'].rolling(
    window=window).mean()
stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()
stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2 * stock_data[
    'Closing_Price'].rolling(window=window).std()

# Calculate RSI for Closing_Price (taking a window of 14 days as an example)
delta = stock_data['Closing_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

# Pivot the stock_dataset to have Date as the index and companies as columns
pivot_stock_data = stock_data.pivot(index='Date',
                                    columns='Name',
                                    values='Closing_Price')

# Calculate correlation matrix
correlation_matrix = pivot_stock_data.corr()

# Calculate quartiles for correlation coefficients
coefficients = correlation_matrix.values.flatten()
lower_quantile = np.percentile(coefficients, 25)
upper_quantile = np.percentile(coefficients, 75)

# Define contours for lower and upper quartiles
contour_lower = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=-1, end=lower_quantile,
                  size=0.05),  # Adjust size as needed
    colorscale='Blues',  # Choose a colorscale for lower quartile contours
    showscale=False,
    name=f'Lower Quartile Contours (-1 - {lower_quantile:.2f})')

contour_upper = go.Contour(
    z=correlation_matrix.values,
    x=correlation_matrix.index,
    y=correlation_matrix.columns,
    contours=dict(start=upper_quantile, end=1,
                  size=0.05),  # Adjust size as needed
    colorscale='Reds',  # Choose a colorscale for upper quartile contours
    showscale=False,
    name=f'Upper Quartile Contours ({upper_quantile:.2f} - 1)')
# Create an empty figure
fig = go.Figure()
fig_rs = go.Figure()
fig_sm = go.Figure()
fig_cs = go.Figure()
fig_bc = go.Figure()
fig_ts = go.Figure()
fig2 = go.Figure()

# Define your layout for each visualization
# Visualization 1 layout
visualization_1_layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown_2',
        options=[{
            'label': company,
            'value': company
        } for company in stock_data['Name'].unique()],
        value=stock_data['Name'].iloc[0]  # Default value for dropdown
    ),
    dcc.Graph(figure=fig2, id='vwap-chart_1')
])

# Combine all visualizations vertically
app.layout = html.Div([
    html.H1("Interactive charts"), visualization_1_layout
])

# Define callback functions for interactivity
# Define callback to update the VWAP chart based on selected company
@app.callback(
    Output('vwap-chart_1', 'figure'),
    [Input('company-dropdown_2', 'value')]
)

def update_fig2_chart(selected_company):
    filtered_stock_data = stock_data[stock_data['Name'] == selected_company]
    fig2 = px.scatter(filtered_stock_data, x='Date', y='VWAP', title=f'Volume-Weighted Average Price (VWAP) for {selected_company}',
                     color='Closing_Price',  # Encode Volume information using color
                     size='Closing_Price',   # Encode Volume information using marker size
                     color_continuous_scale='Viridis',  # Choose a color scale
                     labels={'VWAP': 'VWAP', 'Closing_Price': 'Closing_Price'},  # Set labels for axes
                     hover_data={'Date': '|%B %d, %Y'})  # Format hover data for date
    return fig2

# Run the app
if __name__ == '__main__':
  app.run_server(debug=True, port=int(os.environ.get('PORT', 8050)))