#======================================================================
#15.1 Auto MPG 데이터셋
#======================================================================
import matplotlib.pyplot as plt
import pandas
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#data_url = "https://archive-beta.ics.uci.edu/ml/datasets/auto+mpg"
#dataset_path = keras.utils.get_file("auto-mpg.data", data_url)
#======================================================================
#15.2 데이터준비
#가. 데이터 읽기
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
dataset.tail()
#======================================================================
#나. 누락된 값 확인
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

dataset.isna().sum()
#======================================================================
#나. 누락된 값 제거
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
#======================================================================
#다. 원-핫 인코딩으로 변환
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
#======================================================================
#라. 데이터를 학습셋과 테스트셋으로 분리
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
#======================================================================
#마. 학습 데이터셋의 기술 통계를 확인
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
train_stats
#======================================================================
#바. 데이터셋에서 클래스를 분리
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
#======================================================================
#사. 정규화 작업
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