import numpy as np
import pandas as pd
import os

	
def PreProcess(directory,simdata,valfile,variables):
	#populate train files
	cwd = os.getcwd() 	

	count = 0
	for item in simdata:
		# to specified directory 
		os.chdir(directory) 

		#read the CSV
		data = pd.read_csv(item)
		df = pd.DataFrame()
		for var in variables:
			new = data.loc[data[' name'] == ' '+var[0]][' '+var[1]]
			df[var[0]+var[1]]= new.reset_index(drop=True)
		
		df.dropna()
		count+=1
		os.chdir(cwd)
		df.to_csv('train'+str(count)+'.csv', sep = ',',index=False)

	#populate validation file
	#read the CSV
	os.chdir(directory)
	data = pd.read_csv(valfile)
	df = pd.DataFrame()
	for var in variables:
		new = data.loc[data[' name'] == ' '+var[0]][' '+var[1]]
		df[var[0]+var[1]]= new.reset_index(drop=True)
	
	df.dropna()
	count+=1

	#change to current directory and save file
	os.chdir(cwd)
	df.to_csv('val_data.csv', sep = ',',index=False)

	# back to existing specified directory 
	os.chdir(cwd)


