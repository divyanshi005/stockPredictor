# stockPredictor
Detailed explanation of the code:

	yfinance is a Python library that allows you to download historical market data from Yahoo Finance.
	We then download the required data using yfinance library from the Yahoo database and store it in stockdata variable.
	Data preprocessing: to ensure the data fits within a specific range or distribution.
	Scaling : Adjusting the range of values for a feature to ensure all the values are in the same scale. This is often done by normalizing or standardizing data as some models are sensitive to the scale of input data.
	Common scaling methods: 
	Normalization: Min – Max Scaling: Scales the data to a fixed range (usually [0,1]) by shifting and compressing the original data based on minimum and maximum values of each feature. We generally initialize MinMaxScaler (from sklearn library) with feature_range between 0 and 1 and assign it to a scaler variable, which is later used to fit the scaler to the input data and transform it. 
	FORMULA :  X’= ( X - Xmin )/ ( Xmax – Xmin )
	Standardization: Z Score Scaling: adjusts the data to have a mean of 0 and standard deviation of 1. This is done by first initializing the StandardScaler object from the skearn library and assign it to a variable to later fit and transform the data.
	FORMULA :  X’= (X – µ)/ σ
	Translating: Shifting the data by adding or subtracting a constant value to ensure the data is centered around a specific value (like moving the mean to 0). This is done during standardization.
	After scaling our data, we need to create a dataset of sequential data (time-series data) by  breaking up the larger input dataset into smaller subsequences that can be used for LSTM. In the create_dataset function:
	We loop through the data and create subsequences.
	Append a sequence of length ‘time_step’ to the input. Here, time_step (called window-size or look-back period) is the number of previous data points to look back at when creating the input sequence X.
	Each input sequence x is a window of time_step consecutive data points from the dataset ‘data’. This creates a sliding window over the dataset, capturing the patterns in chunks of time_step.
	Y (the target value ) : the corresponding output y for each sequence is the next value in the dataset,i.e, the value that comes immediately after the windowed input. 
	We are basically trying to predict the next value in the sequence base don the previous time_step values. X and Y are used to train LSTM to predict the next value in a sequence.
	We set the time_step as 100 so input sequences will consist of 100 consecutive data points, and get x and y (input-output pairs) from create_dataset function.
	X is a 2D array where each row represents 100 consecutive data points, and y will be a 1D array containing the corresponding target values (the next data point after the sequence).
	Then we split the data into training and testing ( 80% for training and 20% for testing).
	The training set is used to fit the model, and the test set is used to evaluate the model’s performance to see how well the model generalizes to unseen data.
	We are building a Long Short-Term Memory (LSTM) neural network model using the Keras library, which is a popular type of Recurrent Neural Network (RNN) designed specifically to capture long-term dependencies in sequential data, which traditional RNNs struggle with due to issues like vanishing gradients.
	Before the advent of Transformers, LSTMs were widely used for sequence tasks, including language modeling, machine translation, and time-series forecasting.
	Model initialization : Sequential: This is the Keras model type where you can stack layers one after another in a sequence. It’s useful for building simple feed-forward models or RNNs like LSTMs.
	The LSTM model is compiled and trained on time-series data using the Adam optimizer and mean_squared_error loss function.
	The model's performance is evaluated on test data, and predictions are made.
	The predictions are plotted alongside the actual data to visually compare how well the model has learned to predict stock prices (or another time series).
	Additional predictions are made for the future (based on the last 200 test samples), and the plot is updated to include these predictions.
Details:
	First LSTM Layer: Captures the temporal patterns in the input data and outputs a sequence.
	LSTM(64): This creates an LSTM layer with 64 units (or memory cells). These units learn patterns in the time-series data.
	return_sequences=True: This ensures that the LSTM layer outputs the entire sequence of data, not just the last time step. This is useful when stacking multiple LSTM layers.
	input_shape=(time_step, 1): This defines the shape of the input data. The first value (time_step) represents the number of time steps in each input sequence, and the second value (1) represents the number of features per time step (for univariate data, it is 1).
	When we are working with a time series of stock prices, you may input one stock price per time step (hence the 1).
	Second LSTM Layer : Summarizes the input sequence into a final hidden state:  Another LSTM layer with 64 units. Since this layer does not have return_sequences=True, it only outputs the final hidden state, which summarizes the entire input sequence into a single vector. This final vector is then passed to the next dense layer.
	First Dense Layer: A fully connected layer that further processes the information : A fully connected (Dense) layer with 64 units. This is a feed-forward layer that helps in learning more complex representations after the sequential patterns have been captured by the LSTMs.
	Output Layer: The output layer for making a prediction (e.g., predicting the next value in a time series): The output layer with a single unit (Dense(1)), which is typically used for regression tasks where you're predicting one value (like the next value in a time series). If you're predicting something like a future stock price, the output would be a single continuous value.
MODEL COMPILATION:
	optimizer="adam": Adam (Adaptive Moment Estimation) is a popular and efficient optimization algorithm for training neural networks. It is often used for LSTM models due to its ability to handle large datasets and noisy gradients.
	loss="mean_squared_error": MSE (Mean Squared Error) is used as the loss function. It's appropriate for regression tasks where the goal is to predict continuous values (like stock prices). MSE penalizes larger errors more than smaller errors, which is desirable when predicting time-series data.
Training the Model:
	epochs=10: This means the model will iterate over the entire training dataset 10 times. The more epochs, the more the model learns, but training for too many epochs can cause overfitting.
	batch_size=64: The batch size refers to the number of training samples that are fed into the model at a time before updating the weights. A batch size of 64 means that 64 samples are processed in parallel before the model updates its parameters (this can depend on hardware like GPU/CPU).
Evaluating the Model on Test Data:
	model.evaluate(x_test, y_test): This evaluates the model's performance on the test data (x_test and y_test). It returns the test loss (which is the mean squared error in this case).
	test_loss: The result is printed, giving you the MSE for the test set. A lower loss indicates better performance.
Making Predictions:
	model.predict(x_test): This generates predictions from the model using the test data (x_test). The predictions are in the scaled form (since the data was scaled earlier).
	scaler.inverse_transform(predictions): This step reverses the scaling applied to the predictions, converting them back to the original scale (e.g., the actual stock price). This is necessary to interpret the results in terms of real values.
Preparing Data for Plotting:
	original_data = stockdata["Close"].values: This retrieves the actual stock prices (or the values of the Close column from the stockdata DataFrame).
	predicted_data = np.empty_like(original_data): Creates an empty array (predicted_data) with the same shape as the original_data. This array will hold the predicted stock prices.
	predicted_data[:] = np.nan: This initializes all values in the predicted_data array as NaN (Not a Number). NaN values are used as placeholders to ensure proper alignment of the predicted data.
	predicted_data[-len(predictions):] = predictions.reshape(-1): This fills in the predicted_data array with the model's predictions. It aligns the predictions with the last part of the original data by placing the predictions in the last len(predictions) positions.
Plotting the Original vs. Predicted Data:
	plt.plot(original_data): This plots the original stock prices (the "Close" values).
	plt.plot(predicted_data): This plots the predicted stock prices, making it easy to compare the actual stock prices with the predicted ones.
Making Future Predictions:
	model.predict(x_test[-200:]): This generates predictions for the last 200 data points of x_test. This might be useful if you want to predict future values based on the last few test data points.
	scaler.inverse_transform(new_predictions): This transforms the new predictions back to their original scale.
Appending New Predictions & Plotting:
	predicted_data = np.append(predicted_data, new_predictions): This appends the new predictions (generated in the previous step) to the existing predicted_data. This extends the prediction timeline.
	plt.plot(original_data): This re-plots the original stock prices.
	plt.plot(predicted_data): This re-plots the predicted stock prices, including the newly predicted values.
