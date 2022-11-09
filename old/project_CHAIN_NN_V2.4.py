# -*- coding: utf-8 -*-
'''
Created on Fri Nov  4 13:07:11 2022

@author: João Paulo and Hallvard Bjørgen

NEURAL NETWORK CHAINING FILE
	The idea here is: after using the main network to evaluate our data, 
	we now know which trials the main model predicts the best, so we now make
	a chain of neural networks that will perform binary classification,
	and place first the networks predicting highest accuracy trials, 
	and last the networks predicting the lowest accuracy ones, in an attempt
	to improve their results.
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import matplotlib.pyplot as plt
import os

def get_preprocessed_raw_data(filename):
	try:
		df = pd.read_csv(filename)
		print('-----------------------------------------------')
		print('Preprocessed datafile retrieved successfully...')
		print('-----------------------------------------------')
		return df
	except(FileNotFoundError):
		print(
			'Error handled. File: \'{filename}\' was not found, run the preprocessing first and check spelling of the name.')
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


def make_model(arg_epochs, arg_batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, trial):
	'''
	return: epochs, batch size, accuracy, confusion matrix, time to fit the model
	'''
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


def single_chain(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, trial_no):

	print(f'Processing for trial {trial_no}...')
	trial = 'Trial_'+ f'{trial_no}'
	res, history, model = make_model(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train[trial], X_test, Y_test[trial], X_val, Y_val[trial], trial)
	#Join Training output
	Y_output = model.predict(X_train) 
	Y_output = pd.DataFrame(Y_output.round())
	Y_output = Y_output.rename(columns={0: 'output'})
	X_train = np.array(pd.DataFrame(X_train).join(Y_output))
	#Join Validation output
	Y_output = model.predict(X_val) 
	Y_output = pd.DataFrame(Y_output.round())
	Y_output = Y_output.rename(columns={0: 'output'})
	X_val = np.array(pd.DataFrame(X_val).join(Y_output))
	#Join Test output
	Y_output = model.predict(X_test) 
	Y_output = pd.DataFrame(Y_output.round())
	Y_output = Y_output.rename(columns={0: 'output'})
	X_test = np.array(pd.DataFrame(X_test).join(Y_output))
	return res, history, model, X_train, X_val, X_test
   

def chain_networks(epochs, batch_size, learning_rate,activation_func_hidden, activation_func_output, df_preprocessed, no_of_targets):

	# For storing the outputs
	results = []
	histories = []
	models = []

	# Get train, test and validation data for first iteration
	X_train, Y_train, X_test, Y_test, X_val, Y_val = process_data(df_preprocessed)
	Y_train, Y_test, Y_val = extra_processing(Y_train, Y_test, Y_val)
	Y_train = pd.get_dummies(Y_train)
	Y_test = pd.get_dummies(Y_test)
	Y_val = pd.get_dummies(Y_val)

	sequence = [1,2,3,4,5,6]

	# Run for rest of targets
	for trial_no in sequence:
		res, history, model, X_train, X_val, X_test = single_chain(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, trial_no)
		results.append(res)
		histories.append(history)
		models.append(model)

	return results, histories, models


def latex_out(results, no_of_targets):
	bs = '\\'

	lat_out = open('latex_out.txt', 'w')
	lat_out.write(bs+'begin{figure}\n'+bs+'raggedright\n')
	lat_out.close()

	for i in range(no_of_targets):
		confusion, acc = results[i][3], results[i][2]	
		output = open('latex_out.txt', 'a')
		# Convert to strings to put right allow concatination with rest of string
		tp = confusion[0][0]
		fp = confusion[0][1]
		fn = confusion[1][0]
		tn = confusion[1][1]
		t = tp+fp
		tp_fp = f'{t}'
		fn_tn = f'{fn+tn}'
		tp_fn = f'{tp+fn}'
		fp_tn = f'{fp+tn}'
		n = f'{tp+fp+fn+tn}'

		acc = f'{acc}' 
		trial_no_str = f'{i+1}' 
		output.write(
			bs+'begin{subfigure}{.45'+bs+'textwidth}\n'+bs+'begin{subfigure}{.9'+bs+'textwidth}\n'+bs+'begin{tabular}{l|l|c|c|c}\n'+
			bs+'multicolumn{2}{c}{}&'+bs+'multicolumn{2}{c}{Actual}&'+bs+bs+'\n'+bs+'cline{3-4}\n'+
			bs+'multicolumn{2}{c|}{}&Positive & Negative &'+
			bs+'multicolumn{1}{c}{Total}'+bs+bs+'\n'+
			bs+'cline{2-4}\n'+
			bs+'multirow{2}{*}{'+bs+'rotatebox[origin=c]{90}{'+bs+'parbox[c]{1.3cm}{'+bs+'centering Predicted}}}&Positive &'+f'{tp}'+' & '+f'{fp}'+' & {'+tp_fp+'}'+bs+bs+'\n'+
			bs+'cline{2-4} & Negative & '+f'{fn}'+' & '+f'{tn}'+' & {'+fn_tn+'}'+bs+bs+'\n'+
			bs+'cline{2-4}\n'+
			bs+'multicolumn{1}{c}{} & '+bs+'multicolumn{1}{c}{Total} & '+bs+'multicolumn{1}{c}{'+tp_fn+'}&'+
			bs+'multicolumn{1}{c}{'+fp_tn+'} & '+bs+'multicolumn{1}{c}{'+n+'}\n'+
			bs+'end{tabular}\n'+
			bs+'end{subfigure}\n'+bs+'centering\nAccuracy: '+acc+bs+bs+'\n'+
			bs+'label{fig:trials_conf_acc_'+trial_no_str+'}\n'+
			bs+'caption{Trial '+trial_no_str+'}\n'+
			bs+'end{subfigure}\n')
		print(f"Trial {i} appended to 'latex_out.txt'.")
	output.write(
		bs+'caption{Confusion matrices and accuracies}\n'+
		bs+'label{fig:trials_conf_acc}\n'+
		bs+"end{figure}")
	print("All trials written to 'latex_out.txt'.")


def plot_and_save_graphs(history, trial_no, plot=True, save=True):
	## !!PLOT ALERT!!

	#Data from the last model fitting process
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	#Plotting of Training vs Validation accuracy evolving with epochs
	epochs = range(1, len(acc) + 1)

	fig_acc = plt.figure()

	plt.plot(epochs, acc, 'r', label='Training accuracy')
	plt.plot(epochs, val_acc, 'k', label='Validation accuracy')
	plt.title(f'Trial {trial_no}')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid()
	# Set limits so figures are more intuitive to read next to each other
	plt.ylim([0.6, 1])

	fig_loss = plt.figure()

	plt.plot(epochs, loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'k', label='Validation loss')
	plt.title(f'Trial {trial_no}')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid()
	# Set limits so figures are more intuitive to read next to each other
	plt.ylim([0, 0.20])

	if(plot):
		plt.show()

	if(save):
		# save plots to /figs folder 
		cwd = os.path.abspath(os.getcwd())
		try:
			figs_folder =  os.mkdir(cwd+'/figs')
		except OSError: 
			figs_folder = cwd+'/figs'

		fig_acc.savefig(figs_folder+'/fig_acc_t'+str(trial_no)+'.jpg', bbox_inches='tight', dpi=150)
		fig_loss.savefig(figs_folder+'/fig_loss_t'+str(trial_no)+'.jpg', bbox_inches='tight', dpi=150)
		print(f"Trial {trial_no} saved to /figs")

def main():
	preprocessed_data_filename = 'preprocessed_original_data.csv'
	df_preprocessed = get_preprocessed_raw_data(preprocessed_data_filename)

	activation_func_hidden = 'relu'
	activation_func_output = 'sigmoid'

	epochs = 200
	batch_size = 50
	learning_rate = 1e-3
	no_of_targets = 6

	results, histories, models = chain_networks(epochs, batch_size, learning_rate, activation_func_hidden, activation_func_output, df_preprocessed, no_of_targets)

	# send results to output.txt, latex_out.txt, and plot/save graphs
	latex_out(results, no_of_targets)

	# Create/reset the output file
	# output = open('output.txt', 'w')
	# output.write('OUTPUT\n---------------------------------------')
	# output.close()
	# Add the confusion matriix and accuracy to the output file.
	output = open('output.txt', 'a')
	for i in range(no_of_targets):
		output.write(f'\nTRIAL NUMBER {i+1}\nConfusion matrix:\n{results[i][3]}\nAccuracy: {results[i][2]}\n---------------------------------------\n')
		plot_and_save_graphs(history=histories[i],trial_no=i+1,plot=False,save=True)
	


if __name__ == '__main__':
	   
	time_total = time.time() # For curiosity
	main()
	print(f'Total time elapsed: {round(time.time()-time_total, 1)} seconds.')
