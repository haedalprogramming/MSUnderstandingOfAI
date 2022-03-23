#======================================================================
#10.1 와인 데이터셋
#가. 파일을 업로드
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]
#======================================================================
#나. 판다스로 파일 읽기
#======================================================================
from google.colab import files
uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)
df
#======================================================================
#10.2 데이터 준비
#가. 데이터셋을 분리
#======================================================================
from google.colab import files

uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)


def split_xy(df, class_index):
    dataset = df.values
    X = dataset[:, 0:class_index]
    Y = dataset[:, class_index]
    return X, Y


def split_dataset(df, train_split=0.8, val_split=0.1, seed=0):
    df_sample = df.sample(frac=1, random_state=seed)

    indices_or_sections = [int(train_split * len(df)),
                           int((1 - val_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

import numpy as np

train, val, test = split_dataset(df, seed=seed)

X_train, Y_train = split_xy(train, 12)
X_val, Y_val = split_xy(val, 12)
X_test, Y_test = split_xy(test, 12)
#======================================================================
#나. 속성 스케일 조정
#======================================================================
from google.colab import files

uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)


def split_xy(df, class_index):
    dataset = df.values
    X = dataset[:, 0:class_index]
    Y = dataset[:, class_index]
    return X, Y


def split_dataset(df, train_split=0.8, val_split=0.1, seed=0):
    df_sample = df.sample(frac=1, random_state=seed)

    indices_or_sections = [int(train_split * len(df)),
                           int((1 - val_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

import numpy as np

train, val, test = split_dataset(df, seed=seed)

X_train, Y_train = split_xy(train, 12)
X_val, Y_val = split_xy(val, 12)
X_test, Y_test = split_xy(test, 12)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)
#======================================================================
#10.3 모델 설정
#======================================================================
from google.colab import files

uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)


def split_xy(df, class_index):
    dataset = df.values
    X = dataset[:, 0:class_index]
    Y = dataset[:, class_index]
    return X, Y


def split_dataset(df, train_split=0.8, val_split=0.1, seed=0):
    df_sample = df.sample(frac=1, random_state=seed)

    indices_or_sections = [int(train_split * len(df)),
                           int((1 - val_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

import numpy as np

train, val, test = split_dataset(df, seed=seed)

X_train, Y_train = split_xy(train, 12)
X_val, Y_val = split_xy(val, 12)
X_test, Y_test = split_xy(test, 12)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(32, input_dim=12, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#======================================================================
#10.4 훈련검증, 출력
#======================================================================
from google.colab import files

uploaded = files.upload()
my_data = list(uploaded.keys())[0]

import pandas as pd

df = pd.read_csv(my_data, header=None)


def split_xy(df, class_index):
    dataset = df.values
    X = dataset[:, 0:class_index]
    Y = dataset[:, class_index]
    return X, Y


def split_dataset(df, train_split=0.8, val_split=0.1, seed=0):
    df_sample = df.sample(frac=1, random_state=seed)

    indices_or_sections = [int(train_split * len(df)),
                           int((1 - val_split) * len(df))]

    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)

    return train_ds, val_ds, test_ds


import tensorflow as tf

seed = 0
tf.random.set_seed(seed)

import numpy as np

train, val, test = split_dataset(df, seed=seed)

X_train, Y_train = split_xy(train, 12)
X_val, Y_val = split_xy(val, 12)
X_test, Y_test = split_xy(test, 12)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
X_test = scaler.fit_transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(32, input_dim=12, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=300,
                    batch_size=200)

import matplotlib.pyplot as plt

val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

epochs = range(1, len(val_loss) + 1)

plt.plot(epochs, val_loss, 'bo', label='validation loss')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')

plt.title('Validation loss and accuracy')

plt.show()

print("\n Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))