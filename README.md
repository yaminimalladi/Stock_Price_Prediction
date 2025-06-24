Stock_Price_Prediction_Using_LSTM 
This project demonstrates how to use an LSTM neural network to predict future stock prices using historical data. It utilizes Yahoo Finance data, processes it using MinMax scaling, and trains an LSTM model to predict stock closing prices. The final output includes a visualization comparing the model’s predictions with actual prices and exports the results to CSV.

-Fetches historical stock data from Yahoo Finance using yfinance
-Preprocesses data with MinMaxScaler
-Uses 60-day time windows for LSTM sequence learning
-Builds and trains a 2-layer LSTM model using TensorFlow/Keras
-Evaluates performance using Root Mean Squared Error (RMSE)
-Visualizes actual vs. predicted prices using matplotlib
-Exports predictions to CSV
