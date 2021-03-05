import numpy as np
import tensorflow as tf
import pandas as pd 
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,GRU,Activation
from tensorflow.keras.optimizers import Adam

#cuDnn bug fix
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

class Controller(object):
	def __init__(self,modelname,files,PV,lookback,Default=True):
		self.modelname = modelname
		self.files = files
		self.PV = PV
		self.env_lookback = lookback
		self.TrainPreprocess()
		self.TrainTestSplit()
		#Train and save model if Default is True
		if Default:
			self.TrainCont()
			self.SaveEnv()

	def TrainPreprocess (self):
		#calculate how big the sum of datasets are
		nsamples = 0
		nvariables = 0
		self.InputIndex=[]
		self.PV_max = 0
		self.PV_min = 1
		for file in self.files:
			data = pd.read_csv(file)
			ncol = len(data.columns)
			nsamples += data.shape[0] - self.env_lookback - 2
			nvariables = ncol
			
		#initalize arrays
		self.variables = np.zeros((nsamples,self.env_lookback,nvariables-1), dtype = 'float32')
		self.targets = np.zeros((nsamples,1),dtype = 'float32')
		
		#take data from csv into arrays
		#initalize counter
		self.count = 0
		for file in self.files:
			data = pd.read_csv(file)
			variable_data = data.drop(self.PV,axis = 1)
			self.InputIndex=list(variable_data.columns)
			for t in range(data.shape[0]-self.env_lookback-2):
				self.targets[self.count] = data.at[t+self.env_lookback+1,self.PV]
				self.variables[self.count] = np.asarray(variable_data.iloc[t:t+self.env_lookback,:])
				self.count +=1
	
	def TrainTestSplit (self):
		#calculate seperate 70/20/10 train test val splits
		nx = self.variables.shape[0]
		trainset = int(round(.7*nx,0))
		testset = int(round(.2*nx,0))
		valset = trainset + testset

		#Shuffle the Data and create train test dataframes
		perm = np.random.permutation(nx)
		self.x_train = self.variables[perm[0:trainset]]
		self.y_train = self.targets[perm[0:trainset]]
		self.x_test  = self.variables[perm[trainset:valset]]
		self.y_test = self.targets[perm[trainset:valset]]
		self.x_val  = self.variables[perm[valset:nx]]
		self.y_val = self.targets[perm[valset:nx]]


	def TrainCont(self,fc1_dims=16,fc2_dims=16,lr=.001,ep=1000):

		# clear previous model just in case
		tf.keras.backend.clear_session()

		self.model = Sequential([
			Dense(fc1_dims, input_shape=(self.x_train.shape[1],self.x_train.shape[2])), #,return_sequences=True
			Activation('tanh'),
			Dense(fc2_dims),
			Activation('relu'),
			Dense(1),
			Activation('linear')])

		self.model.compile(optimizer=Adam(lr=lr), loss='mse')
		print(self.model.summary())
		self.model.fit(self.x_train, self.y_train, 
          				batch_size=500, epochs=ep, verbose=1, 
						validation_data = (self.x_test,self.y_test))
		
		score = self.model.evaluate(self.x_val, self.y_val, verbose=0)
		self.SaveEnv()
		print('Model Saved.  Valadation mean squared error ',score)
		

	def SaveEnv(self):
		#create unique string to save model to working directory
		self.model.save(self.modelname +'.h5')

		with open(self.modelname +'.txt', 'w') as filehandle:
			filehandle.write('%s\n' % 'lookback=')
			filehandle.write('%s\n' % self.env_lookback)
			filehandle.write('%s\n' % 'PV=')
			filehandle.write('%s\n' % self.PV)
			filehandle.write('%s\n' % '***Input Index***')
			for listitem in self.InputIndex:
				filehandle.write('%s\n' % listitem)
		filehandle.close()
