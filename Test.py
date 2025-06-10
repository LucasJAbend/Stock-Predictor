import pandas as pd
import pytz
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# NEW IMPORTS FOR AI/ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# NEW IMPORT FOR GRIDSEARCHCV
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE # Ensure this is also imported

# --- Data Acquisition ---
stock = input("Enter a stock symbol: ").upper()

# Download the data with pre and post-market data
# Changed period from '10d' to '7d' to respect yfinance 1m data limits
# Using 30d and 5m as per your last run
df = yf.download(tickers=stock, period='30d', interval='5m', prepost=True)

# Handle potential empty dataframe if download fails
if df.empty:
    print(f"No data downloaded for {stock}. Please check the symbol or period.")
    exit()

df.index = df.index.tz_convert('US/Eastern')
# Flatten MultiIndex columns
df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

print(df.head())
print(df.columns)

# Standardize column names for easier access if yfinance includes the ticker symbol
if f'Close_{stock}' in df.columns:
    CLOSE_COL = f'Close_{stock}'
    HIGH_COL = f'High_{stock}'
    LOW_COL = f'Low_{stock}'
    OPEN_COL = f'Open_{stock}'
    VOLUME_COL = f'Volume_{stock}'
else: # Assume direct column names
    CLOSE_COL = 'Close'
    HIGH_COL = 'High'
    LOW_COL = 'Low'
    OPEN_COL = 'Open'
    VOLUME_COL = 'Volume'

# --- Feature Engineering ---
# Calculate VWAP
df['Typical Price'] = (df[HIGH_COL] + df[LOW_COL] + df[CLOSE_COL]) / 3
# Ensure initial Cumulative TPV and Cumulative Volume are not NaN for the first few rows
df['Cumulative TPV'] = (df['Typical Price'] * df[VOLUME_COL]).cumsum()
df['Cumulative Volume'] = df[VOLUME_COL].cumsum()
df['VWAP'] = df['Cumulative TPV'] / df['Cumulative Volume']

# Calculate Simple Moving Averages
df['SMA_9'] = df[CLOSE_COL].rolling(window=9).mean()
df['SMA_20'] = df[CLOSE_COL].rolling(window=20).mean()

# Calculate Exponential Moving Averages
df['EMA_9'] = df[CLOSE_COL].ewm(span=9, adjust=False).mean()
df['EMA_12'] = df[CLOSE_COL].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df[CLOSE_COL].ewm(span=26, adjust=False).mean()

# Calculate MACD and Signal line
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD Histogram'] = df['MACD'] - df['Signal Line']

# Calculate buying and selling volume
df['Buy Volume'] = np.where(df[CLOSE_COL] > df[CLOSE_COL].shift(1), df[VOLUME_COL], 0)
df['Sell Volume'] = np.where(df[CLOSE_COL] < df[CLOSE_COL].shift(1), df[VOLUME_COL], 0)

# --- Define Categorical Target Variable for "Next Hour" ---

# Calculate the percentage change for the *next hour* (60 minutes / 5 min interval = 12 steps)
df['Next_Hour_Close'] = df[CLOSE_COL].shift(-12) # Shift by 12 steps for 60 minutes
df['Price_Pct_Change_Next_Hour'] = ((df['Next_Hour_Close'] - df[CLOSE_COL]) / df[CLOSE_COL]) * 100

# Define the thresholds for categorization for HOURLY changes (tune these carefully!)
# Based on previous discussions, these thresholds might need further adjustment for optimal performance
THRESHOLD_STRONG_UP = 0.7
THRESHOLD_UP = 0.3
THRESHOLD_NEUTRAL_RANGE = 0.1
THRESHOLD_DOWN = -0.3
THRESHOLD_STRONG_DOWN = -0.7

def categorize_price_change_hourly(pct_change):
    if pd.isna(pct_change):
        return np.nan
    elif pct_change >= THRESHOLD_STRONG_UP:
        return 'Strong Up'
    elif pct_change > THRESHOLD_UP:
        return 'Up'
    elif abs(pct_change) <= THRESHOLD_NEUTRAL_RANGE:
        return 'Neutral'
    elif pct_change < THRESHOLD_STRONG_DOWN:
        return 'Strong Down'
    elif pct_change <= THRESHOLD_DOWN:
        return 'Down'
    else: # Fallback for any edge cases not covered
        return np.nan

df['Target_Category_String'] = df['Price_Pct_Change_Next_Hour'].apply(categorize_price_change_hourly)

# Map string categories to numerical labels
category_mapping = {
    'Strong Down': 0,
    'Down': 1,
    'Neutral': 2,
    'Up': 3,
    'Strong Up': 4
}
df['Target_Category_Numerical'] = df['Target_Category_String'].map(category_mapping)

# Define feature columns to use for the model
features = [
    OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL,
    'VWAP', 'SMA_9', 'SMA_20', 'EMA_9', 'EMA_12', 'EMA_26',
    'MACD', 'Signal Line', 'MACD Histogram', 'Buy Volume', 'Sell Volume'
]

# Drop rows with NaN values (from shifts or rolling window calculations)
feature_and_target_cols = features + ['Target_Category_Numerical']
for col in feature_and_target_cols:
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found in DataFrame. Please check feature list or calculations.")

df.dropna(subset=feature_and_target_cols, inplace=True)

print("\nDataFrame with new target variable for Next Hour prediction (after NaN drop):")
print(df[['Price_Pct_Change_Next_Hour', 'Target_Category_String', 'Target_Category_Numerical']].head())
print(df[['Price_Pct_Change_Next_Hour', 'Target_Category_String', 'Target_Category_Numerical']].tail())
print(f"\nDistribution of Target Categories for Next Hour Prediction:\n{df['Target_Category_String'].value_counts()}")

# --- AI Model Preparation and Training ---
print("\n--- Preparing Data for AI Model ---")

X = df[features]
y = df['Target_Category_Numerical']

print(f"Number of samples (rows) available for training: {len(X)}")
print(f"Number of features (columns): {len(features)}")

# Data Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print("\nFeatures scaled successfully.")

# Time-Series Splitting (CRITICAL for Time Series Data)
train_ratio = 0.8
train_size = int(len(X_scaled_df) * train_ratio)

X_train, X_test = X_scaled_df[:train_size], X_scaled_df[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"First training sample index: {X_train.index.min()} | Last training sample index: {X_train.index.max()}")
print(f"First testing sample index: {X_test.index.min()} | Last testing sample index: {X_test.index.max()}")

# --- Apply SMOTE for Class Imbalance ---
print("\n--- Addressing Class Imbalance with SMOTE ---")

# Before SMOTE
print(f"Original training target distribution: {Counter(y_train)}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# After SMOTE
print(f"Resampled training target distribution: {Counter(y_train_resampled)}")
print("Class imbalance addressed in training data.")

# --- Model Training: Random Forest Classifier with GridSearchCV ---
print("\n--- Training Random Forest Classifier with GridSearchCV for Hyperparameter Tuning ---")

# Define the parameter grid to search
param_grid = {
    'n_estimators': [500, 700, 1000], # Number of trees
    'max_depth': [10, 20, 10, 20, 50, None],     # Maximum depth of each tree (None for full depth)
    'min_samples_split': [2, 5, 10],     # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 3, 5, 10],      # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'] # Number of features to consider when looking for the best split
}

# Initialize a fresh Random Forest Classifier for GridSearchCV
# Set class_weight='balanced' here to consider it during tuning, or rely solely on SMOTE
rf_initial = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf_initial,
    param_grid=param_grid,
    scoring='f1_weighted', # Optimize for weighted F1-score to consider class imbalance in evaluation
    cv=3,                   # 3-fold cross-validation
    n_jobs=-1,              # Use all available CPU cores
    verbose=2               # Print progress updates
)

# Fit GridSearchCV to the RESAMPLED training data
grid_search.fit(X_train_resampled, y_train_resampled)

print("\n--- GridSearchCV Results ---")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best F1-weighted score from cross-validation: {grid_search.best_score_:.4f}")

# Use the best model found by GridSearchCV
best_rf_model = grid_search.best_estimator_

print("\nRandom Forest model (tuned) trained successfully!")

# --- Model Evaluation with Best Model ---
print("\n--- Evaluating Best Random Forest Model ---")

y_pred_rf = best_rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy on Test Set: {accuracy_rf:.4f}")

target_names = [name for name, num in sorted(category_mapping.items(), key=lambda item: item[1])]
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for Best Random Forest Classifier ({stock})')
plt.show()

print("\nFeature Importances (Top 10 - Best Model):")
feature_importances = pd.Series(best_rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances.head(10))


# --- Plotly Visualization (Optional) ---
# You can comment out this entire section or just fig.show() if you want to focus on console output

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2],
                    vertical_spacing=0.1,
                    subplot_titles=(f'{stock} Live Share Price with VWAP, SMAs, and EMAs',
                                    'Volume', 'MACD'))

fig.add_trace(go.Candlestick(x=df.index,
                             open=df[OPEN_COL],
                             high=df[HIGH_COL],
                             low=df[LOW_COL],
                             close=df[CLOSE_COL],
                             name='Market Data'),
              row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name='VWAP', line=dict(color='orange', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_9'], name='9-period SMA', line=dict(color='blue', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='20-period SMA', line=dict(color='purple', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], name='9-period EMA', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='12-period EMA', line=dict(color='red', width=1, dash='dash')), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df['Buy Volume'], name='Buy Volume', marker_color='green'), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['Sell Volume'], name='Sell Volume', marker_color='red'), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue', width=1)), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['Signal Line'], name='Signal Line', line=dict(color='red', width=1)), row=3, col=1)
fig.add_trace(go.Bar(x=df.index, y=df['MACD Histogram'], name='MACD Histogram', marker_color='green'), row=3, col=1)

fig.update_layout(
    title=f'{stock} Live Share Price with VWAP, SMAs, EMAs, Volume, and MACD (Including Pre/Post-Market)',
    yaxis_title='Stock Price (USD per Share)',
    xaxis_title='Time',
    barmode='relative',
    showlegend=True,
    legend=dict(itemclick="toggle", itemdoubleclick="toggleothers")
)

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
fig.show()