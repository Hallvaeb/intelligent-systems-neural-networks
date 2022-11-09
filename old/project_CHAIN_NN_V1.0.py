# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:07:11 2022

@author: João Paulo
"""

#NEURAL NETWORK CHAINING FILE

#The idea here is: after using the main network to evaluate our data, 
#we now know which trials the main model predicts the best, so we now make
#a chain of neural networks that will perform binary classification,
#and place first the networks predicting highest accuracy trials, 
#and last the networks predicting the lowest accuracy ones, in an attempt
#to improve their results.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import time
import os
def get_preprocessed_raw_data(filename):
    try:
        df = pd.read_csv(filename)
        print("Preprocessed datafile retrieved successfully...")
        return df
    except(FileNotFoundError):
        print(
            "Error handled. File: \"{filename}\" was not found, run the preprocessing first and check spelling of the name.")
        exit()
    # And if different error we want it to raise whatever error happens..
def process_data(df_final):

    #df_final = df_final.drop(columns=['inf_ecg.csv','inf_gsr.csv','inf_ppg.csv','pixart.csv'])
    seed_train_test = 0
    seed_test_val = 0
    # random state is a seed value
    train_data = df_final.sample(frac=0.75, random_state=seed_train_test)
    test_data = df_final.drop(train_data.index)
    Y_train = train_data[['target']]
    X_train = train_data.drop(columns=['target'])
    y_test = test_data[['target']]
    x_test = test_data.drop(columns=['target'])
    X_val, X_test, Y_val, Y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=seed_test_val)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_val = scaler.fit_transform(X_val)

    X_train = np.asarray(X_train).astype(np.float32)
    Y_train = np.asarray(Y_train).astype(int)
    X_test = np.asarray(X_test).astype(np.float32)
    Y_test = np.asarray(Y_test).astype(int)
    X_val = np.asarray(X_val).astype(np.float32)
    Y_val = np.asarray(Y_val).astype(int)

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def chain_networks(epochs, batch_size, learning_rate,activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val):
	
    Y_train = pd.get_dummies(Y_train)
    Y_test = pd.get_dummies(Y_test)
    Y_val = pd.get_dummies(Y_val)
    
    res1, history1, model1 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_1'], X_test, Y_test['Trial_1'], X_val, Y_val['Trial_1'], 'Trial_1')
    #Join Training output
    Y_output = model1.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model1.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model1.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    res2, history2, model2 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_2'], X_test, Y_test['Trial_2'], X_val, Y_val['Trial_2'], 'Trial_2')
    #Join Training output
    Y_output = model2.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model2.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model2.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    res3, history3, model3 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_3'], X_test, Y_test['Trial_3'], X_val, Y_val['Trial_3'], 'Trial_3')
    #Join Training output
    Y_output = model3.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model3.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model3.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    res4, history4, model4 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_4'], X_test, Y_test['Trial_4'], X_val, Y_val['Trial_4'], 'Trial_4')
    #Join Training output
    Y_output = model4.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model4.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model4.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    res5, history5, model5 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_5'], X_test, Y_test['Trial_5'], X_val, Y_val['Trial_5'], 'Trial_5')
    #Join Training output
    Y_output = model5.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model5.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model5.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    res6, history6, model6 = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train['Trial_6'], X_test, Y_test['Trial_6'], X_val, Y_val['Trial_6'], 'Trial_6')
    #Join Training output
    Y_output = model6.predict(X_train) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_train = np.array(pd.DataFrame(X_train).join(Y_output))
    #Join Validation output
    Y_output = model6.predict(X_val) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_val = np.array(pd.DataFrame(X_val).join(Y_output))
    #Join Test output
    Y_output = model6.predict(X_test) 
    Y_output = pd.DataFrame(Y_output.round())
    Y_output = Y_output.rename(columns={0: 'output'})
    X_test = np.array(pd.DataFrame(X_test).join(Y_output))
    
    return res1, res2, res3, res4, res5, res6

def make_model(arg_epochs, arg_batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, trial):
	"""
	return: epochs, batch size, accuracy, confusion matrix, time to fit the model
	"""
	n_features = X_train.shape[1]
	# Define model
	model = Sequential()
	model.add(Dense(40, activation=activation_func_hidden, kernel_initializer='he_normal', input_shape=(n_features,)))
	model.add(Dropout(0.2))
	model.add(Dense(20, activation=activation_func_hidden, kernel_initializer='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(20, activation=activation_func_hidden, kernel_initializer='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation=activation_func_output))
	#model.add(Dense(1, activation=None))

	# Define the optimizer
	model.compile(
		optimizer=Adam(learning_rate),
		loss=MeanSquaredError(),
		metrics=['accuracy']
	)

	start_time = time.time()
	# Fit the model
	history  = model.fit(X_train, Y_train, epochs=arg_epochs, batch_size=arg_batch_size, verbose=1, validation_data=(X_val, Y_val))
	time_to_fit = round(time.time()-start_time, 2)

	## Testing Data
	y_pred = model.predict(X_test)
    
	conf = confusion_matrix(Y_test, y_pred.round())
	acc = round(accuracy_score(Y_test, y_pred.round()), 5)

	res = arg_epochs, arg_batch_size, acc, conf, time_to_fit
	return res, history, model
def extra_processing(Y_train, Y_test, Y_val):
    
    new_y_train = pd.DataFrame(Y_train)
    new_y_test = pd.DataFrame(Y_test)
    new_y_val = pd.DataFrame(Y_val)
        
    new_y_train = new_y_train.rename(columns={0: 'Trial'})
    new_y_test = new_y_test.rename(columns={0: 'Trial'})
    new_y_val = new_y_val.rename(columns={0: 'Trial'})
    
    Y_train_new = pd.get_dummies(new_y_train, columns=['Trial'])
    Y_test_new = pd.get_dummies(new_y_test, columns=['Trial'])
    Y_val_new = pd.get_dummies(new_y_val, columns=['Trial'])
    
    return Y_train_new, Y_test_new, Y_val_new

preprocessed_data_filename = "preprocessed_original_data.csv"
df_preprocessed = get_preprocessed_raw_data(preprocessed_data_filename)
X_train, Y_train, X_test, Y_test, X_val, Y_val = process_data(df_preprocessed)

Y_train, Y_test, Y_val = extra_processing(Y_train, Y_test, Y_val)

activation_func_hidden = 'relu'
activation_func_output = 'sigmoid'

epochs = 200
batch_size = 50
learning_rate = 1e-3
        
res1, res2, res3, res4, res5, res6 = chain_networks(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val)

print(res1[3],'', res2[3],'',res3[3],'',res4[3],'',res5[3],'',res6[3], sep=os.linesep)
print(res1[2],'', res2[2],'',res3[2],'',res4[2],'',res5[2],'',res6[2], sep=os.linesep)


