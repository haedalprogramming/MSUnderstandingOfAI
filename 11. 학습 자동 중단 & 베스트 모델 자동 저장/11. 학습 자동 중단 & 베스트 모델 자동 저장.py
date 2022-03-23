#======================================================================
#11.1 모델 파일 저장 경로 설정
#가. 베스트 모델을 자동 저장할 폴더와 파일명 설정
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

import os

MODEL_PATH = './model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

model_path = MODEL_PATH + '{epoch:03d} - {val_loss:.4f}.h5'
#======================================================================
#나. 자동저장 콜백함수를 정의
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

import os

MODEL_PATH = './model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

model_path = MODEL_PATH + '{epoch:03d} - {val_loss:.4f}.h5'

from tensorflow.keras.callbacks import ModelCheckpoint

cb_checkpointer = ModelCheckpoint(
    filepath = model_path,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)
#======================================================================
#다. 자동 중단을 위한 콜백함수를 설정
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

import os

MODEL_PATH = './model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

model_path = MODEL_PATH + '{epoch:03d} - {val_loss:.4f}.h5'

from tensorflow.keras.callbacks import ModelCheckpoint

cb_checkpointer = ModelCheckpoint(
    filepath = model_path,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)

from tensorflow.keras.callbacks import EarlyStopping

cb_early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 100
)
#======================================================================
#라. 모델 새로 설정
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

import os

MODEL_PATH = './model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

model_path = MODEL_PATH + '{epoch:03d} - {val_loss:.4f}.h5'

from tensorflow.keras.callbacks import ModelCheckpoint

cb_checkpointer = ModelCheckpoint(
    filepath = model_path,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)

from tensorflow.keras.callbacks import EarlyStopping

cb_early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 100
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model12 = Sequential()

model12.add(Dense(32, input_dim=12, activation='relu'))
model12.add(Dense(16, activation='relu'))
model12.add(Dense(4, activation='relu'))
model12.add(Dense(1, activation='sigmoid'))

model12.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
#======================================================================
#마. callbacks 매개변수에 콜백함수를 등록
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

import os

MODEL_PATH = './model/'

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

model_path = MODEL_PATH + '{epoch:03d} - {val_loss:.4f}.h5'

from tensorflow.keras.callbacks import ModelCheckpoint

cb_checkpointer = ModelCheckpoint(
    filepath = model_path,
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True
)

from tensorflow.keras.callbacks import EarlyStopping

cb_early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 100
)

model12 = Sequential()

model12.add(Dense(32, input_dim=12, activation='relu'))
model12.add(Dense(16, activation='relu'))
model12.add(Dense(4, activation='relu'))
model12.add(Dense(1, activation='sigmoid'))

model12.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

history = model12.fit(X_train,
                      Y_train,
                      validation_data=(X_val, Y_val),
                      epochs = 500,
                      batch_size = 200,
                      callbacks = [cb_checkpointer, cb_early_stopping],
                      verbose = 0)
