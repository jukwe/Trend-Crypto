import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf #accessing yahoo finace since datareader is old
from datetime import datetime #for handling date time
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras  
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, LSTM


# fetching stock data
def get_ticker_data(ticker : str) -> pd.DataFrame:

    try:
        # Dynamic date range
        start = datetime(2016, 1, 1)
        end = datetime.now().strftime('%Y-%m-%d') # make more dynamic to auto update to current
        #df = data.DataReader('TSLA', 'yahoo', start, end)
        df = yf.download(ticker, start, end) # using yahoo finance to access data
        df = df.reset_index()
        # df.tail()
        # df = df.drop(['Date', 'Adj Close'], axis = 1)
        return df
    except Exception as e:
        print(f'An error occurred while fetching data for {ticker}: {e}')

def train_model(df : pd.DataFrame, data_train : pd.DataFrame, scaler : MinMaxScaler) -> np.array:
    arr_train = scaler.fit_transform(data_train)

    #splitting Data into x_train and y_train to train model
    x_train = []
    y_train = []

    # print(arr_train.shape[0])
    for i in range(100, arr_train.shape[0]):
        x_train.append(arr_train[i-100: i]) #using the previous 100 day data to determine the following days price
        y_train.append(arr_train[i, 0]) # the 101st days price 

    x_train, y_train = np.array(x_train), np.array(y_train) #converted to numpy array
    print(x_train.shape, y_train.shape)

    return x_train, y_train

def make_model(x_train : np.array, y_train : np.array):
    try:
        model = Sequential() # initializing the type of model
        model.add(LSTM(units= 50, activation= 'relu', return_sequences= True, input_shape= (x_train.shape[1], x_train.shape[2],1))) # adding first layer with ipu
        model.add(Dropout(0.2))

        model.add(LSTM(units= 60, activation= 'relu', return_sequences= True))
        model.add(Dropout(0.3))

        model.add(LSTM(units= 80, activation= 'relu', return_sequences= True))
        model.add(Dropout(0.4))

        model.add(LSTM(units= 120, activation= 'relu', return_sequences= True))
        model.add(Dropout(0.5))


        model.add(Dense(units= 1)) #connects all the layers of the model


        model.compile(optimizer= 'adam', loss= 'mean_squared_error')
        model.fit(x_train, y_train, epochs = 25)

        #saving model
        model.save('crypto_model.keras')
        return model
    except Exception as e:
        print(f"An error occurred while creating or training the model: {e}")

def test_model(past_100_days : pd.DataFrame, data_test : pd.DataFrame, scaler : MinMaxScaler) -> np.array:
    #Testing part
    
    #use .concat instead of .append because .append was removed

    final_df = pd.concat([past_100_days, data_test], ignore_index= True) # resetting the index to avoid conflicts with indexing of DataFrame final_df

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])


    x_test, y_test = np.array(x_test), np.array(y_test)
    print(x_test.shape, y_test.shape)

    return x_test, y_test

def make_prediction(loaded_model, x_test : np.array, y_test : np.array , scaler : MinMaxScaler):
    #Making Predictions
    y_predict = loaded_model.predict(x_test)

    scaler = scaler.scale_

    scale = 1/scaler[0]
    y_predict = y_predict * scaler
    y_test = y_test * scale

    print(x_test.shape)
    return y_predict, y_test

if __name__ == "__main__":
    # Collecting what Crypto the user wants to view
    st.title('Cryptocurrency Trend Prediction')
    ticker = st.text_input('Enter Crypto Ticker (EX: \'BTC-USD\')', 'BTC-USD')

    #getting dataframe of data
    df = get_ticker_data(ticker)

    # Describing Data
    st.subheader('Data from 2016 to Now')
    st.write(df.describe()) #table
    st.dataframe(df)

    # Create a candlestick chart
    df['Date'] = pd.to_datetime(df['Date'])
    df['20wma'] = df['Close'].rolling(window=140).mean()

    fig = go.Figure(data=[go.Candlestick(
        x = df['Date'],
        open = df['Open'],
        high = df['High'],
        low = df['Low'],
        close = df['Close'],
        name = "Candlesticks",
    )])

    fig.add_trace(go.Scatter(
        x = df['Date'],
        y = df['20wma'],
        line = dict(color = '#e0e0e0'),
        name = '20wk MA'
    ))

    # Set chart layout and title
    fig.update_layout(
        title = f"{ticker} - Year-to-Date Candlestick Chart",
        xaxis_title = 'Date',
        yaxis_title = 'Price',
        # xaxis_rangeslider_visible = False,
        template = 'plotly_dark',
        width = 800,
        height = 600
    )
    # fig.update_yaxes(type='log')

    # Display the chart using Streamlit
    st.write(fig)

    scaler = MinMaxScaler(feature_range= (0,1)) #data is scaled down to between 0 and 1

    # Splitting data into testing and training
    data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #train with 70% of data
    data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])#test with 30% of data

    x_train, y_train = train_model(df, data_train, scaler)

    # model = make_model(x_train, y_train)

    # to load model
    model = load_model('crypto_model.keras')
    past_100_days = data_train.tail(100)

    x_test, y_test = test_model(past_100_days, data_test, scaler)

    y_predict, y_test = make_prediction(model, x_test, y_test, scaler)

    #Final Graph
    st.header('Predictions vs Original')
    fig_2 = plt.figure(figsize= (12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predict, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig_2)