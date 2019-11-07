import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

Dataset = pd.read_csv('./2017EE10436.csv',header=None);
num_features = 784
F = Dataset.iloc[:,:num_features].values
s = StandardScaler()
F = s.fit_transform(F)
T = Dataset.iloc[:,-1].values
T = to_categorical(T)

F_train, F_test, T_train, T_test = train_test_split(F, T, test_size=0.20,random_state=100)

model = Sequential()
model.add(Dense(64, input_dim = F_train.shape[1], activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(F_train, T_train, epochs=70, batch_size=32, validation_data=(F_test, T_test), verbose = 0)
score = model.evaluate(F_test, T_test, batch_size=32, verbose = 2)
print(score)