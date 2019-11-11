import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

Dataset = pd.read_csv('./2017EE10433.csv',header=None);
F = Dataset.iloc[:,:-1].values
s = StandardScaler()
F = s.fit_transform(F)
T = Dataset.iloc[:,-1].values
T = to_categorical(T)

F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)
epochs = 150
batch_size = 16

model = Sequential()
model.add(Dense(125, input_dim = F_train.shape[1], activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

sgd = tensorflow.keras.optimizers.SGD(lr=0.5,nesterov=True)
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,optimizer='sgd', metrics=['accuracy'])
model.fit(F_train, T_train, epochs = epochs, batch_size = batch_size, validation_data=(F_test, T_test), verbose = 1)
score = model.evaluate(F_test, T_test, verbose = 2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])