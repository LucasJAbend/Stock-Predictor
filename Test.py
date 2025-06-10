import pandas as pd
import pytz
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Get the stock symbol
stock = input("Enter a stock symbol: ").upper()

# Download the data with pre and post-market data
df = yf.download(tickers=stock, period='1d', interval='1m', prepost=True)

df.index = df.index.tz_convert('US/Eastern')
# Flatten MultiIndex columns
df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

# After downloading the data
print(df.head())
print(df.columns)

# Required columns with ticker suffix
required_columns = [f'High_{stock}', f'Low_{stock}', f'Close_{stock}', f'Volume_{stock}']
if not all(col in df.columns for col in required_columns):
    print("One or more required columns are missing.")
    exit()

# Calculate VWAP
df['Typical Price'] = (df[f'High_{stock}'] + df[f'Low_{stock}'] + df[f'Close_{stock}']) / 3
df['Cumulative TPV'] = (df['Typical Price'] * df[f'Volume_{stock}']).cumsum()
df['Cumulative Volume'] = df[f'Volume_{stock}'].cumsum()
df['VWAP'] = df['Cumulative TPV'] / df['Cumulative Volume']

# Calculate Simple Moving Averages
df['SMA_9'] = df[f'Close_{stock}'].rolling(window=9).mean()
df['SMA_20'] = df[f'Close_{stock}'].rolling(window=20).mean()

# Calculate Exponential Moving Averages
df['EMA_9'] = df[f'Close_{stock}'].ewm(span=9, adjust=False).mean()
df['EMA_12'] = df[f'Close_{stock}'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df[f'Close_{stock}'].ewm(span=26, adjust=False).mean()

# Calculate MACD and Signal line
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD Histogram'] = df['MACD'] - df['Signal Line']

# Calculate buying and selling volume
df['Buy Volume'] = np.where(df[f'Close_{stock}'] > df[f'Close_{stock}'].shift(1), df[f'Volume_{stock}'], 0)
df['Sell Volume'] = np.where(df[f'Close_{stock}'] < df[f'Close_{stock}'].shift(1), df[f'Volume_{stock}'], 0)

# Print the data for reference
print(df)

# Get stock info
info = yf.Ticker(stock)
print(info.info)

# Create the figure with subplots
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    row_heights=[0.6, 0.2, 0.2],
                    vertical_spacing=0.1,
                    subplot_titles=(f'{stock} Live Share Price with VWAP, SMAs, and EMAs', 
                                    'Volume', 'MACD'))

# Add the candlestick chart
fig.add_trace(go.Candlestick(x=df.index,
                             open=df[f'Open_{stock}'],
                             high=df[f'High_{stock}'],
                             low=df[f'Low_{stock}'],
                             close=df[f'Close_{stock}'], 
                             name='Market Data'),
              row=1, col=1)

# Add indicators
fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_9'], name='9-period SMA', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20-period SMA', line=dict(color='purple', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='9-period EMA', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='12-period EMA', line=dict(color='red', width=1, dash='dash')), row=1, col=1)

# Add buy/sell volumes
fig.add_trace(go.Bar(x=df.index, y=df['Buy Volume'], name='Buy Volume', marker_color='green'), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['Sell Volume'], name='Sell Volume', marker_color='red'), row=2, col=1)

# Add MACD and signal line
fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal Line', line=dict(color='red', width=1)), row=3, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['MACD Histogram'], name='MACD Histogram', marker_color='green'), row=3, col=1)

# Layout config
fig.update_layout(
    title=f'{stock} Live Share Price with VWAP, SMAs, EMAs, Volume, and MACD (Including Pre/Post-Market)',
    yaxis_title='Stock Price (USD per Share)',
    xaxis_title='Time',
    barmode='relative',
    showlegend=True,
    legend=dict(itemclick="toggle", itemdoubleclick="toggleothers")
)

# Add range slider and buttons
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    ),
    row=1, col=1
)
#edit
# Show the chart
fig.show()
