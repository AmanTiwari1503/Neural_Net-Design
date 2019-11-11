from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from mlxtend.data import loadlocal_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
s = StandardScaler()

# =============================================================================
# Dataset = pd.read_csv('./2017EE10436.csv',header=None);
# F = Dataset.iloc[:,:-1].values
# T = Dataset.iloc[:,-1].values
# F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)
# 
# =============================================================================
#loading train data
F_train, T_train = loadlocal_mnist(images_path='./MNIST_data/train-images.idx3-ubyte', labels_path='./MNIST_data/train-labels.idx1-ubyte')

#Loading test data
F_test, T_test = loadlocal_mnist(images_path='./MNIST_data/test-images.idx3-ubyte', labels_path='./MNIST_data/test-labels.idx1-ubyte')

batch_size = 2000
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    F_train = F_train.reshape(F_train.shape[0], 1, img_rows, img_cols)
    F_test = F_test.reshape(F_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    F_train = F_train.reshape(F_train.shape[0], img_rows, img_cols, 1)
    F_test = F_test.reshape(F_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

F_train = F_train.astype('float32')
F_test = F_test.astype('float32')
F_train = F_train/255
F_test = F_test/255
print(F_train.shape[0], 'train samples')
print(F_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
T_train = to_categorical(T_train, num_classes)
T_test = to_categorical(T_test, num_classes)

classifier = Sequential()
classifier.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu', input_shape = input_shape))
classifier.add(Conv2D(64, kernel_size=(3, 3), padding = 'same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(num_classes, activation='softmax'))

sgd = SGD(lr = 0.01,nesterov=True)
classifier.compile(loss = categorical_crossentropy, optimizer = sgd, metrics=['accuracy'])

#classifier.summary()

hist = classifier.fit(F_train, T_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_data=(F_test, T_test))

activation_model = models.Model(inputs=classifier.input, outputs=[layer.output for layer in classifier.layers[:-1]])
img_imp = np.expand_dims(F_test[0], axis=0)
images_per_row = 8
activations = activation_model.predict(img_imp) 	

layer_names = []
for layer in classifier.layers[:-1]:
	layer_names.append(layer.name)

for layer_name, layer_activation in zip(layer_names, activations): 
	n_features = layer_activation.shape[-1]
	size = layer_activation.shape[1]
	n_cols = n_features // images_per_row
	display_grid = np.zeros((size * n_cols, images_per_row * size))
	for col in range(n_cols):
		for row in range(images_per_row):
			channel_image = layer_activation[0,:,:,col * images_per_row + row]
			channel_image -= channel_image.mean() 
			channel_image /= channel_image.std()
			channel_image *= 64
			channel_image += 128
			channel_image = np.clip(channel_image, 0, 255).astype('uint8')
			display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
	scale = 1. / size
	plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
	plt.title(layer_name)
	plt.grid(False)
	plt.imshow(display_grid, aspect='auto', cmap='viridis')



acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

plt.plot(range(epochs), acc, 'bo', label='Training acc')
plt.plot(range(epochs), val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc = 'best')
plt.figure()
plt.plot(range(epochs), loss, 'bo', label='Training loss')
plt.plot(range(epochs), val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc = 'best')
plt.show()

score = classifier.evaluate(F_test, T_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])