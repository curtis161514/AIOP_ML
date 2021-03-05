from AIOP.ImitationController import Controller
import numpy as np
from AIOP.DataProcessor import PreProcess

##################################################################
#----------create 2d dataframes from sim output-------------------
##################################################################
#directory the raw data is stored
directory = '/home/curtiswright/Documents/AI-OP/Boiler/Boiler_Data'
#name of data files
simdata = ['session_16.csv','session_17.csv','session_21.csv','session_40.csv',
			'session_169.csv','session_244.csv','session_246.csv']
valfile = 'session_247.csv'	

variables = [	['FIC-1100','sRaw'],
				['FIC-1101','sMV'],
				['TI-1100','sRaw']
									]

PreProcess(directory,simdata,valfile,variables)

####################################################################
#---------Initalize some common variables shared btw classes--------
####################################################################
#process data to learn from
files = ['train1.csv','train2.csv','train3.csv','train4.csv',
			'train5.csv','train6.csv','train7.csv']
#name to save the trained model to
env_modelname = 'FIC1100_Controller'
#process response variable
PV = 'FIC-1100sRaw' 


##################################################################
#----------------Build and Train Controller Model------------
###################################################################

lookback = 3 #time required to capture environment dynamics
controller = Controller(env_modelname,files,PV,lookback,Default_Env=False)

#train controller model
controller.TrainCont(fc1_dims=16,fc2_dims=16,lr=.001,ep=50)

