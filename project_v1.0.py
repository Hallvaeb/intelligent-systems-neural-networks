import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, accuracy_score
import time
import os

FILENAMES = ["inf_ecg.csv", "inf_gsr.csv", "inf_ppg.csv", 'pixart.csv', "NASA_TLX.csv"]

def process_df(df, filename, person_no, df_pro, datasplit = 10):
	"""
	
	"""
	no_avg_rows = round(len(df)/datasplit)

	# Section the data into less rows and average the sections into 1 row each.
	for datasplit_no in range(datasplit):
		section = df[no_avg_rows*datasplit_no:no_avg_rows*(datasplit_no+1)]
		sum_section = section.sum(numeric_only=True, axis=0)
		avg_section = sum_section/no_avg_rows

		# Splitting the columns. We have 6 columns of data from 6 different tests.
		tot_no_of_tests = 6
		for test_no in range(0, tot_no_of_tests):
			# Don't ask me to explain the below equation to finding the offset please. It works.
			offset = (datasplit_no*tot_no_of_tests+test_no) + (person_no*datasplit*tot_no_of_tests)
			df_pro.at[offset, 'person_no'] = int(person_no) # WHY NO INT?
			df_pro.at[offset, filename] = avg_section[test_no]
			df_pro.at[offset, 'target'] = test_no+1
			
def get_df(filename, path_to_data, person_no):
	# Run same procedure for every person in the dataset. 
	# for person_no in range (2, 25):
		# Find the correct pre-numbers for the datafile name.
	if(person_no < 10):
		preno_filenames = "00"
	else:
		preno_filenames = "0"

	# Convert to string and append the pre-numbers
	tmp = "% s" % person_no
	p_no_str = preno_filenames+tmp
	path_to_file = path_to_data+p_no_str+"/"+filename

	# Reading data
	if filename != "NASA_TLX.csv":
		try:
			df = pd.read_csv(path_to_file)
			return df
		except:
			return pd.DataFrame()
	else: 
		try:
			df_tlx = pd.read_csv(path_to_file)
			df_tlx = df_tlx.iloc[0:6,0:7]
			return df_tlx
		except:
			return pd.DataFrame()
	
def process_df_tlx(df_tlx, filename, person_no, df_tlx_pro, datasplit = 10):
	
	# Splitting the columns. We have 6 columns of data from 6 different tests.
	tot_no_of_tests = 6
	for datasplit_no in range(datasplit):
		for test_no in range(0, tot_no_of_tests):
			#Don't ask me to explain the below equation to finding the offset please. It works.
			offset = (datasplit_no*tot_no_of_tests+test_no) + (person_no*datasplit*tot_no_of_tests)
			count = 0
			for column in df_tlx_pro.columns:                    
				df_column = df_tlx.loc[count]
				df_tlx_pro.at[offset, column] = df_column[test_no+1]
				count += 1
				
def process_data(df_final, df_target):
	
	#Join features and targets for train/test/validation data split
	df_final = df_final.merge(df_target)
	#df_final = df_final.drop(columns=['inf_ecg.csv','inf_gsr.csv','inf_ppg.csv','pixart.csv'])
	seed_train_test = 0
	seed_test_val = 0
	train_data = df_final.sample(frac=0.75,random_state=seed_train_test) #random state is a seed value
	test_data = df_final.drop(train_data.index)
	Y_train = train_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
	X_train = train_data.drop(columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6']) 
	y_test = test_data[['out1', 'out2', 'out3', 'out4', 'out5', 'out6']]
	x_test = test_data.drop(columns=['out1', 'out2', 'out3', 'out4', 'out5', 'out6'])
	X_val, X_test, Y_val, Y_test = train_test_split(x_test, y_test,test_size=0.4, random_state = seed_test_val)
	X_val = np.array(X_val)
	Y_val = np.array(Y_val)
	X_test = np.array(X_test)
	Y_test = np.array(Y_test)
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	
	#Normalize the data
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

def print_results(epochs, batch_size, acc, conf, time_to_fit):
	'''
	Prints the values given from the model
	'''
	print('---------------------')
	print('Epochs:', epochs)
	print('Batch size:', batch_size)
	print('Training Confusion Matrix:', conf)
	print('Accuracy:', acc)
	print('Time to fit:', round(time_to_fit), "seconds")

def make_model(arg_epochs, arg_batch_size, learning_rate, activation_func_hidden, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val):
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
	model.add(Dense(6, activation=activation_func_output))
	#model.add(Dense(1, activation=None))

	# Define the optimizer
	model.compile(
		optimizer=Adam(learning_rate),
		loss=CategoricalCrossentropy(),
		metrics=['accuracy']
	)

	start_time = time.time()
	# Fit the model
	history  = model.fit(X_train, Y_train, epochs=arg_epochs, batch_size=arg_batch_size, verbose=1, validation_data=(X_val, Y_val))
	time_to_fit = round(time.time()-start_time, 2)

	## Testing Data
	y_pred = model.predict(X_test)
	
	conf = multilabel_confusion_matrix(Y_test, y_pred.round())
	acc = round(accuracy_score(Y_test, y_pred.round()), 5)

	res = arg_epochs, arg_batch_size, acc, conf, time_to_fit
	return res, history, model

def run_once(epochs, batch_size, learning_rate, activation_func_1, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores):
	"""
	Creates one model of given parameters
	"""
	results = []
	accuracies = []
	res, history, model = make_model(epochs, batch_size, learning_rate, activation_func_1, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val)
	results.append(res)
	accuracies.append(res[2])
	sum = 0
	for acc in accuracies:
		sum += acc
	acc_avg = sum/len(accuracies)
			
	for res in results:
		print_results(res[0], res[1], res[2], res[3], res[4])
	  
	print("!!!!! ALL MODEL FITTINGS ARE FINISHED !!!!!")
	print(f"Average accuracy for {epochs} epochs, {batch_size} batch size and {learning_rate} LR was: {acc_avg}")
	
	# Adding to compare different models. res [4] is the time to fit. About the same for all so using last is fine.
	avg_scores.append([epochs, batch_size, learning_rate, acc_avg, res[4]])
	return history, model

def encode_target(df_final):
	df_final_t = df_final['target']
	df_final_target = pd.DataFrame({'out1':[], 'out2':[], 'out3':[], 'out4':[], 'out5':[], 'out6':[]})
	for i in range(0, len(df_final_t)):
		if df_final_t.iloc[i] == 1:
			df_final_target.loc[i] = [1, 0, 0, 0, 0, 0]
		elif df_final_t.iloc[i] == 2:
			df_final_target.loc[i] = [0, 1, 0, 0, 0, 0]
		elif df_final_t.iloc[i] == 3:
			df_final_target.loc[i] = [0, 0, 1, 0, 0, 0]
		elif df_final_t.iloc[i] == 4:
			df_final_target.loc[i] = [0, 0, 0, 1, 0, 0]
		elif df_final_t.iloc[i] == 5:
			df_final_target.loc[i] = [0, 0, 0, 0, 1, 0]
		elif df_final_t.iloc[i] == 6:
			df_final_target.loc[i] = [0, 0, 0, 0, 0, 1]
		else:
			raise('Check Data Type')
	return df_final_target
 
def main():

   	# Create processed data dataframe on the heap to fill it later.
	df_pro = pd.DataFrame({'person_no':[], 'inf_ecg.csv':[], 'inf_gsr.csv':[], 'inf_ppg.csv':[], 'pixart.csv':[], 'target':[]})
	df_tlx_pro = pd.DataFrame({'Mental Demand':[], 'Physical Demand':[], 'Temporal Demand':[], 'Performance':[], 'Effort':[], 'Frustration':[]})
	
	cwd = os.path.abspath(os.getcwd())
	path_TLX =  cwd+"/MAUS/Subjective_rating/"    
	path_to_data = cwd+"/MAUS/Data/Raw_data/" 

   	#We want to first select the file to include. The algorithm should gather data from all participants everytime anyways.
	for filename in FILENAMES:
		for person_no in range(0, 30): 
			if filename != "NASA_TLX.csv":
				df = get_df(filename, path_to_data, person_no)
				if(len(df.any()) > 0):
					process_df(df, filename, person_no, df_pro, datasplit = 10)
			else:
				df_tlx = get_df(filename, path_TLX, person_no)                
				if(len(df_tlx.any()) > 0):
					process_df_tlx(df_tlx, filename, person_no, df_tlx_pro, datasplit=10)
					df_tlx = get_df(filename, path_TLX, person_no)        
	df_pro = df_pro.reset_index(drop=True)
	df_tlx_pro = df_tlx_pro.reset_index(drop=True)
	df_final = df_pro.join(df_tlx_pro)
		
	df_target = encode_target(df_final)
	df_final = df_final.drop('target', axis = 1)
	df_final = df_final.join(df_target)
	df_final = df_final.drop('person_no', axis = 1)
	X_train, Y_train, X_test, Y_test, X_val, Y_val = process_data(df_final, df_target)

	#Activation functions for the network
	activation_func_1 = 'relu'
	activation_func_output = 'softmax'

	# For saving the average scores of different types of runs.
	avg_scores = []
	
	history, model = run_once(5, 200, 1e-4, activation_func_1, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
	history, model = run_once(5, 200, 1e-4, activation_func_1, activation_func_output, X_train, Y_train, X_test, Y_test, X_val, Y_val, avg_scores)
	
	print(f"Average accuracies for the different runs [epochs, batch size, learning rate, average accuracy, time to fit]:{avg_scores}")

if __name__ == "__main__":
	   
   	main()