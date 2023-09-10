import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file into a pandas dataframe
filename='unit1.csv'
df = pd.read_csv(filename)
# Clean dataset: 
# Drop columns that are not directly informative for our LSTM model
if f'Unnamed: {20}' in df.columns:
    df.drop(columns=['Unnamed: 20'],inplace=True)
df.drop(columns=['wind_plant', 'date_from', 'date_to', 'utc', 'latitude', 'longitude', 'Turbine Model', 'Forecast(mwh)', 'Production Year'], inplace=True)

# Convert 'Limitation to MW' - if 'no' then 0, if float then it x equals that
df['Limitation to MW'] = df['Limitation to MW'].apply(lambda x: 0 if isinstance(x, str) and x.lower() == 'no' else x)
# df.dropna(subset=['Real Prod(mwh)'], inplace=True)
# Splitting the data into train and test based on 'Real Prod(mwh)' column
train_df = df[df['Real Prod(mwh)'].notna() & (df['Real Prod(mwh)'] != 0)]
test_df = df[df['Real Prod(mwh)'].isna()]
# Separating the target variable
y_train = train_df['Real Prod(mwh)'].values
X_train = train_df.drop(columns=['Real Prod(mwh)'])
X_test = test_df.drop(columns=['Real Prod(mwh)'])
# Normalize the data for LSTM
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshaping the data for LSTM: (samples, time steps, features)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=150, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)
model.save('trained_model.h5')
# Load the test data (assuming 'Real Prod(mwh)' is missing or 0)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Make predictions on the test data
test_predictions = model.predict(X_test_reshaped)
nan_indices = df[df['Real Prod(mwh)'].isna()].index
df = pd.read_csv(filename)
# Fill NaN values in 'Real Prod(mwh)' with values from the predictions at corresponding indices
for i, idx in enumerate(nan_indices):
    df.at[idx, 'Real Prod(mwh)'] = "{:.3f}".format(test_predictions[i][0])

output_csv_file = 'out_'+filename
df.to_csv(output_csv_file, index=False)

print(test_predictions, len(test_predictions))
# df['Real Prod(mwh)'].fillna(pd.Series(test_predictions), inplace=True)
# Print the test predictions
print("Test Predictions (Real Prod(mwh)):")
print(test_predictions.size)
print(test_predictions[0])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
