import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 784
F = Dataset.iloc[:,:num_features].values
s = StandardScaler()
F = s.fit_transform(F)
T = Dataset.iloc[:,-1].values

def sigmoid(X):
	for i in range(len(X)):
		X[i] = 1/(1 + np.exp(-X[i]))
	return X

def ReLU(X):
	for i in range(len(X)):
		if X[i] < 0:
			X[i] = 0
	return X

def softmax(X):
	a = np.exp(X)
	return a/np.sum(a)


def one_hot_encoder(index,n_inputs=10):
	X = np.zeros((n_inputs,1))
	X[index]=1
	return X

def one_hot(X):
	a = np.max(X)
	for i in range(len(X)):
		if X[i] == a:
			X[i] = 1
		else:
			X[i] = 0
	return X

def sig_derivative(X):
	return np.multiply(X,(1-X))


def relu_derivative(X):
	X[X>0] = 1
	return X

class layer(object):
	def __init__(self,input_units,output_units,act_fn):
		self.input_units = input_units
		self.output_units = output_units
		self.act_fn = act_fn
		self.b = np.random.rand(self.output_units,1)
		self.w = np.random.rand(self.input_units,self.output_units)
		self.gradient = None
	
	def calculate_output(self,layer_input):
		self.layer_input = layer_input.reshape((-1,1))
		self.param1 = np.add(np.matmul(self.w.T,self.layer_input),self.b)
		if self.act_fn is 'sigmoid':
			self.output = sigmoid(self.param1)
		elif self.act_fn is 'ReLU':
			self.output = ReLU(self.param1)
		elif self.act_fn is 'softmax':
			self.output = softmax(self.param1)
#			print(a)
#			self.output = one_hot_encoder(a)
		else:
			raise Exception('Invalid activation function')

if __name__ == "__main__":
	def neural_net_initialise(n_inputs,n_hidden_units,n_outputs):
		network = list()
		ob_layer = layer(n_inputs,n_hidden_units[0],'sigmoid')
		network.append(ob_layer)
		for i in range(len(n_hidden_units)-1):
			ob_layer = layer(n_hidden_units[i],n_hidden_units[i+1],'sigmoid')
			network.append(ob_layer)
		ob_layer = layer(n_hidden_units[-1],n_outputs,'softmax')
		network.append(ob_layer)
		return network
	
	
	def forward_propogation(network,input_row):
		inputs = input_row
		for n_layer in network:
			n_layer.calculate_output(inputs)
			inputs = n_layer.output
	
	
	def backward_propogation(network,output_row):
		actual_output = output_row.reshape((-1,1))
		for i in reversed(range(len(network))):
			if i == len(network)-1:
				network[i].gradient = np.subtract(network[i].output,actual_output)
			else:
				if network[i].act_fn is 'sigmoid':
					der = sig_derivative(network[i].param1)
				elif network[i].act_fn is 'ReLU':
					der = relu_derivative(network[i].param1)
				else:
					raise Exception('Invalid activation function')
				network[i].gradient = np.multiply(np.matmul(network[i+1].w,network[i+1].gradient),der)
	
	def weight_update(network,learn_rate):
		for n_layer in network:
			weight_up = learn_rate*np.matmul(n_layer.layer_input,n_layer.gradient.T)
			n_layer.w = np.subtract(n_layer.w,weight_up)
			n_layer.b = n_layer.gradient
	
	def neural_net_train(X_train,Y_train,epochs,learn_rate,n_inputs,n_outputs,n_hidden_units):
		net = neural_net_initialise(n_inputs,n_hidden_units,n_outputs)
		for i in range(epochs):
			print('Epoch number = '+str(i))
			for j in range(X_train.shape[0]):
				forward_propogation(net,X_train[j])
				backward_propogation(net,one_hot_encoder(Y_train[j],n_outputs))
				weight_update(net,learn_rate)
		return net
	
	def determine_accuracy():
		F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)
		n = neural_net_train(F_train,T_train,300,1e-1,F_train.shape[1],10,[180])
		label_pred = np.zeros(F_test.shape[0])
		for i in range(F_test.shape[0]):
			forward_propogation(n,F_test[i])
			label_pred[i],j = np.unravel_index(n[-1].output.argmax(), n[-1].output.shape)
			print(int(label_pred[i]),end = ' ')
		print('\n')
		accuracy_test = metrics.accuracy_score(T_test.T,label_pred)
		print('Test accuracy is = '+str(accuracy_test))
		
	determine_accuracy()