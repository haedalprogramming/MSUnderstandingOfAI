#======================================================================
#9.1 데이터셋
#가. 데이터셋 업로드하기
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]
#======================================================================
#나. 데이터셋 읽어 보기
#======================================================================
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print('훈련 데이터의 크기: ',train_data.shape)
print('테스트 데이터의 크기: ',test_data.shape)
#======================================================================
#9.2 데이터 준비
#가. 데이터 변환
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

e.classes_
#======================================================================
#나. 클래스를 인코딩
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

Y
#======================================================================
#다. 학습셋과 테스트셋으로 분리한다
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)
#======================================================================
#9.3 모델 구성
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#======================================================================
#9.4 훈련 검증
#가. 모델을 학습
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=5,
                    validation_data=(X_test, Y_test))
#======================================================================
#나. history에 값을 저장하고 loss 분석
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=5,
                    validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#======================================================================
#다. history에 값을 저장하고 정확도 분석
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=5,
                    validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
#======================================================================
#라. 과적합 해결결
#=====================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)

dataset = df.values
X = dataset[:,0:60].astype(float)
Y_obj = dataset[:,60]

from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=seed)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=60, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=20,
                    batch_size=5,
                    validation_data=(X_test, Y_test))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

test_mae_score, test_mae_score = model.evaluate(X_test, Y_test)
test_mae_score