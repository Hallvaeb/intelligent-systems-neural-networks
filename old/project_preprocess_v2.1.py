import pandas as pd
import os

FILENAMES = ["inf_ecg.csv", "inf_gsr.csv", "inf_ppg.csv", 'pixart.csv', "NASA_TLX.csv"]

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
				
def process_df(df, filename, person_no, df_pro, datasplit = 10):
	no_avg_rows = round(len(df)/datasplit)
	# Section the data into less rows and average the sections into 1 row each.
	for datasplit_no in range(datasplit):
		section = df[no_avg_rows*datasplit_no:no_avg_rows*(datasplit_no+1)]
		sum_section = section.sum(numeric_only=True, axis=0)
		avg_section = sum_section/no_avg_rows

		# Splitting the columns. We have 6 columns of data from 6 different tests.
		n_columns = 6
		for test_no in range(0, n_columns):
			offset = (datasplit_no*n_columns+test_no) + (person_no*datasplit*n_columns)
			if filename == "NASA_TLX.csv":
				count = 0
				for column in ['Mental Demand', 'Physical Demand', 'Temporal Demand', 'Performance', 'Effort', 'Frustration']:                    
					df_column = df.loc[count]
					df_pro.at[offset, column] = df_column[test_no+1]
					count += 1
			else:
				df_pro.at[offset, 'person_no'] = int(person_no) # WHY NO INT?
				df_pro.at[offset, filename] = avg_section[test_no]
				df_pro.at[offset, 'target'] = test_no+1
	
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
 
def create_preprocessed_raw_data(output_filename):
	   	# Create processed data dataframe on the heap to fill it later.
	df_pro = pd.DataFrame({'person_no':[], 'inf_ecg.csv':[], 'inf_gsr.csv':[], 'inf_ppg.csv':[], 'pixart.csv':[], 'Mental Demand':[], 'Physical Demand':[], 'Temporal Demand':[], 'Performance':[], 'Effort':[], 'Frustration':[], 'target':[]})
	
	cwd = os.path.abspath(os.getcwd())
	path_TLX =  cwd+"/MAUS/Subjective_rating/"    
	path_to_data = cwd+"/MAUS/Data/Raw_data/" 

	print("Getting and processing each file... This should take about 10 seconds...")
   	#We want to first select the file to include. The algorithm should gather data from all participants everytime anyways.
	for filename in FILENAMES:
		for person_no in range(0, 30): 
			if filename != "NASA_TLX.csv":
				path = path_to_data
			else:
				path = path_TLX
			df = get_df(filename, path, person_no)
			if(len(df.any()) > 0):
				process_df(df, filename, person_no, df_pro, datasplit = 10)
	print("All files processed!")

	df_pro = df_pro.reset_index(drop=True)
	df_target = encode_target(df_pro)
	df_pro = df_pro.drop('target', axis = 1)
	df_pro = df_pro.join(df_target)
	df_pro = df_pro.drop('person_no', axis = 1)
	str = df_pro.to_csv(output_filename, index=False)
	print("pandas:", str)
    
def main():

	output_filename = "preprocessed_raw_data.csv"

	create_preprocessed_raw_data(output_filename)

	print(f"Data preprocessed. Check the file: \"{output_filename}\"")
	print(f"If file name already existed, nothing has happened. Remove your old file or change the name in main of this file.")

if __name__ == "__main__":
	
   	main()