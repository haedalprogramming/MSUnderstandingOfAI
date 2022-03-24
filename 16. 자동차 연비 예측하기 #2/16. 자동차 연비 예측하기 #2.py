#======================================================================
#16.1 모델 설정
#가. 모델 설게
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()
#======================================================================
#나. 모델 요약
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

model.summary()
#======================================================================
#다. 1000번 훈련, 기록
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot()])
#======================================================================
#라. history에 저장된 통계치를 사용해 모델의 훈련과정을 시각화1
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
#======================================================================
#마. history에 저장된 통계치를 사용해 모델의 훈련과정을 시각화2
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize = (8, 12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)
#======================================================================
#마. 자동으로 멈추도록 콜백 만들기
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [early_stop, PrintDot()])

plot_history(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize = (8, 12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)
#======================================================================
#바. 평균의 절대오차 확인
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [early_stop, PrintDot()])

plot_history(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize = (8, 12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label = 'Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label = 'Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data,
                                test_labels,
                                verbose = 2)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
#======================================================================
#사. MPG값을 예측
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [early_stop, PrintDot()])

plot_history(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_=plt.plot([-100, 100], [-100, 100])
#======================================================================
#사. 오차의 분포를 히스토그램으로 살펴보자
#======================================================================
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Mdoel Year', 'Origin']
#raw_dataset = pd.read_csv(dataset_path, names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
raw_dataset = pd.read_csv("auto-mpg.data", names = column_names, na_values = "?", comment = '\t', sep=" ", skipinitialspace= True)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = keras.Sequential([
      layers.Dense(64, activation = 'relu',
                   input_shape=[len(train_dataset.keys())]),
      layers.Dense(64, activation = 'relu'),
      layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss = 'mse',
                  optimizer = optimizer,
                  metrics = ['mae', 'mse'])
    return model

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss',patience = 10)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [early_stop, PrintDot()])

plot_history(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

test_predictions = model.predict(normed_test_data).flatten()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_= plt.ylabel("Count")

