import numpy as np
import pandas as pd

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 784
F = Dataset.iloc[:,:num_features].values
T = Dataset.iloc[:,-1].values

def sigmoid(X):
	return 1.0/(1.0 + np.exp(-X))

def ReLU(X):
	for i in range(len(X)):
		if X[i] < 0:
			X[i] = 0
	return X

class layer(object):
	def __init__(self,input_units,output_units,act_fn):
		self.input_units = input_units
		self.output_units = output_units
		self.act_fn = act_fn
		self.b = np.random.rand(self.output_units,1)
		self.w = np.random.rand(self.input_units,self.output_units)
	
	def calculate_output(self,layer_input):
		layer_input = layer_input.reshape((self.input_units,1))
		self.param1 = np.add(np.matmul(self.w.T,layer_input),self.b)
		if self.act_fn is 'sigmoid':
			self.output = sigmoid(self.param1)
		elif self.act_fn is 'ReLU':
			self.output = ReLU(self.param1)
		else:
			raise Exception('Invalid activvation function')
		 
						  
		