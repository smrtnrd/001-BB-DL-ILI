import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input, concatenate
from keras.models import Model

def create_model(self):
    for i in range(self.n_inputs):
        inputName = "{}_input".format(i)

        lstm_input = keras.layers.Input(
            shape=(self.timesteps, self.features),
            batch_shape=(self.batch_size, self.timesteps, self.features),
            name=inputName)
        self.lstmInputs.append(lstm_input)

        lstm_layer = LSTM(self.n_neurons, 
                        return_sequences=False, 
                        stateful=True,
                        batch_input_shape=(self.batch_size, self.timesteps, self.features))(self.lstmInputs[i])
        self.lstmLayers.append(lstm_layer)

    #combined the output
    output = keras.layers.concatenate(self.lstmLayers)
    output = Dense(1, activation='relu',
                    name='wheighthedAverage_output')(output)
    stateInput = self.lstmInputs
    model = keras.models.Model(inputs=stateInput, outputs=[output])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model

# evaluate the model on a dataset, returns RMSE in transformed units
def evaluate(model, raw_data, scaled_dataset, scaler, offset, batch_size):
	# separate
	X, y = scaled_dataset[:,0:-1], scaled_dataset[:,-1]
	# reshape
	reshaped = X.reshape(len(X), 1, 1)
	# forecast dataset
	output = model.predict(reshaped, batch_size=batch_size)
	# invert data transforms on forecast
	predictions = list()
	for i in range(len(output)):
		yhat = output[i,0]
		# invert scaling
		yhat = invert_scale(scaler, X[i], yhat)
		# invert differencing
		yhat = yhat + raw_data[i]
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_data[1:], predictions))
	return rmse


    def create_model(self):
        for i in range(self.n_inputs):
            inputName = "{}_input".format(i)

            lstm_input = keras.layers.Input(
                shape=(self.timesteps, self.features),
                batch_shape=(self.batch_size, self.timesteps, self.features),
                name=inputName)
            self.lstmInputs.append(lstm_input)

            lstm_layer = LSTM(self.n_neurons,
                            return_sequences=False, 
                            stateful=True,
                            batch_input_shape=(self.batch_size, self.timesteps, self.features))(self.lstmInputs[i])
            self.lstmLayers.append(lstm_layer)

        #combined the output
        output = keras.layers.concatenate(self.lstmLayers)
        output = Dense(1, activation='relu',
                       name='wheighthedAverage_output')(output)
        stateInput = self.lstmInputs
        model = keras.models.Model(inputs=stateInput, outputs=[output])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        print(model.summary())
        return model


def create_model(labels, features, timesteps = 1, batch_size = 52, n_neurons = 50):
    #dataLength =  4 weeks
    stateInputs = {}
    stateLayers = []
    i = 0
    for label in labels:
        inputName = "{}_input".format(label)
        stateInputs[inputName] = Input(shape=(timesteps,features),
                                       batch_shape =(batch_size, timesteps, features), 
                                       name=inputName)
    for state in stateInputs:
        stateL = LSTM(n_neurons, return_sequences=False, stateful=True,
                            batch_input_shape=(batch_size, timesteps, features))(stateInputs[state])
        stateLayers.append(stateL)
    #combined the output
    output = keras.layers.concatenate(stateLayers)
    output = Dense(1, activation='relu', name='wheighthedAverage_output')(output)
    stateInput = stateInputs.values()
    
    model = Model(inputs = stateInput, outputs = [output])
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

def get_tt_data(datasets, n_total_years = 260, n_train_weeks = 156 ):
    train_features, train_label = list(), list()
    test_features, test_label = list(), list()
    for data in datasets:
        values = data.head(n_total_years).values
        train = values[:n_train_weeks, :]
        test = values[n_train_weeks:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:,-1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        train_features.append(train_X)
        train_label.append(train_y)
        test_features.append(test_X)
        test_label.append(test_y)
    print("number of weeks in a year: {}".format(n_total_years))
    print("number of weeks in the training set : {}".format(n_train_weeks))
    return train_features, train_label, test_features, test_label

# fit an LSTM network to training data
def fit_lstm(train, test, raw, scaler, batch_size, nb_epoch, neurons):
    
	train_features, train_label, test_features, test_label = get_tt_data(temp_reframed, n_total_years, n_train_weeks )
	# prepare model
    model = create_model(states_label, features, timesteps = 1, batch_size = 52, n_neurons = 50)
    # fit model
	train_rmse, test_rmse = list(), list()
	for i in range(nb_epoch):
		model.fit(
                    train_features,
                    train_label[0],  # label for the targeted state
                    validation_data=(
                        test_features,
                        test_label[0]),
                    epochs=2000,
                    verbose=0,
                    shuffle=False,
                    batch_size=52,
                    callbacks=[checkpoint],
                    initial_epoch=0
                )
		model.reset_states()
		# evaluate model on train data
		raw_train = raw[-(len(train)+len(test)+1):-len(test)]
		train_rmse.append(evaluate(model, raw_train, train, scaler, 0, batch_size))
		model.reset_states()
		# evaluate model on test data
		raw_test = raw[-(len(test)+1):]
		test_rmse.append(evaluate(model, raw_test, test, scaler, 0, batch_size))
		model.reset_states()
	history = DataFrame()
	history['train'], history['test'] = train_rmse, test_rmse
	return history

