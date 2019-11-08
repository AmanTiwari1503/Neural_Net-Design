import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
np.set_printoptions(threshold=40)

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 784
F = Dataset.iloc[:,:num_features].values
s= StandardScaler()
F = s.fit_transform(F)
T = Dataset.iloc[:,-1].values

F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)

def sigmoid(X):
	return 1.0/(1.0 + np.exp(-X))

def ReLU(X):
	X[X < 0] = 0
	return X

def softmax(X):
	a = np.exp(X)
	return a/np.sum(a, axis=0, keepdims=True)


def one_hot_encoder(Y,n_inputs=10):
	X = np.zeros((n_inputs,len(Y)))
	for i in range(X.shape[1]):
		for j in range(X.shape[0]):
			if j == Y[i] :
				X[j][i] = 1
	return X
	

#def one_hot(X):
#	a = np.max(X)
#	for i in range(len(X)):
#		if X[i] == a:
#			X[i] = 1
#		else:
#			X[i] = 0
#	return X

def sig_derivative(X):
	K = sigmoid(X)
	return np.multiply(K,(1-K))


def relu_derivative(X):
	X[X>0] = 1
	return X

def error(Y,T):
	X = np.zeros(Y.shape[1])
	for i in range(Y.shape[1]):
		X[i] = -np.sum(np.multiply(T[:,i],np.log(Y[:,i])))
	return X

class layer(object):
	def __init__(self,input_units,output_units,act_fn):
		self.input_units = input_units
		self.output_units = output_units
		self.act_fn = act_fn
		self.b = np.random.rand(self.output_units,1)*0.1
		self.w = np.random.rand(self.input_units,self.output_units)*0.1
		self.gradient = None
	
	def calculate_output(self,layer_input):
		self.layer_input = layer_input
		self.param1 = np.add(np.matmul(self.w.T,self.layer_input),self.b)
		if self.act_fn is 'sigmoid':
			self.output = sigmoid(self.param1)
		elif self.act_fn is 'ReLU':
			self.output = ReLU(self.param1)
		elif self.act_fn is 'softmax':
			self.output = softmax(self.param1)
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
	
	
	def forward_propogation(network,input_matrix):
		inputs = input_matrix
		for n_layer in network:
			n_layer.calculate_output(inputs)
			inputs = n_layer.output
	
	
	def backward_propogation(network,output_row,batch_size):
		actual_output = output_row
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
	
	def weight_update(network,learn_rate,batch_size):
		for n_layer in network:
			weight_up = (learn_rate)*np.matmul(n_layer.layer_input,n_layer.gradient.T)/batch_size
			n_layer.w = np.subtract(n_layer.w,weight_up)
			n_layer.b -= learn_rate*np.sum(n_layer.gradient, axis = 1, keepdims = True)/batch_size
	
	def neural_net_train(X_train,Y_train,epochs,learn_rate,n_inputs,n_outputs,n_hidden_units,batch_size):
		net = neural_net_initialise(n_inputs,n_hidden_units,n_outputs)
		for i in range(epochs):
			print('Epoch number = '+str(i))
			batches = create_batches(X_train,Y_train.reshape((-1,1)),batch_size)
			for batch in batches:
				X_m,Y_m = batch
				forward_propogation(net,X_m.T)
				backward_propogation(net,one_hot_encoder(Y_m,n_outputs),batch_size)
				weight_update(net,learn_rate,batch_size)
		return net
	
	def create_batches(X, Y, batch_size):			#for creating batches
		batches = []
		data = np.column_stack((X,Y))
#		np.random.shuffle(data)
		n_batches = data.shape[0]//batch_size
		i = 0

		for i in range(n_batches):
			batch = data[i * batch_size:(i + 1)*batch_size, :]
			X_m = batch[:, :-1]
			Y_m = batch[:, -1]
			batches.append((X_m, Y_m))
		i = i+1
		if data.shape[0] % batch_size != 0:
			batch = data[i * batch_size:data.shape[0]]
			X_m = batch[:, :-1]
			Y_m = batch[:, -1]
			batches.append((X_m, Y_m))
		return batches
	
	def train_neural_net(X_train,Y_train,epochs = 100,learning_rate = 6e-2,hidden_units = [100],batch_size = 32):
		n = neural_net_train(X_train,Y_train,epochs,learning_rate,X_train.shape[1],10,hidden_units,batch_size)
		return n
		
	def test_neural_test(net,X_test,Y_test):
		label_pred = np.zeros(X_test.shape[0])
		for i in range(X_test.shape[0]):
			forward_propogation(net,X_test[i].reshape((-1,1)))
			label_pred[i],j = np.unravel_index(net[-1].output.argmax(), net[-1].output.shape)
		accuracy_test = metrics.accuracy_score(Y_test.T,label_pred)
		return accuracy_test
	
	def hyperparameter_variation(X_train,Y_train,X_test,Y_test,param,param_var):
		param_array = []
		if param is "epochs":
			for i in param_var:
				model = train_neural_net(X_train,Y_train,epochs = i)
				param_array.append(test_neural_test(model,X_test,Y_test))
		elif param is "learning_rate":
			for i in param_var:
				model = train_neural_net(X_train,Y_train,learning_rate = i)
				param_array.append(test_neural_test(model,X_test,Y_test))
		elif param is "batch_size":
			for i in param_var:
				model = train_neural_net(X_train,Y_train,batch_size = i)
				param_array.append(test_neural_test(model,X_test,Y_test))
		elif param is "hidden_units":
			for i in param_var:
				model = train_neural_net(X_train,Y_train,hidden_units = i)
				param_array.append(test_neural_test(model,X_test,Y_test))
		else:
			raise Exception("Invalid hyperparameter")
		return param_array
	