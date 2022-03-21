#======================================================================
# 4.3 코드 작성
#1. 데이터셋 불러오기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#======================================================================
#2. 신경망 구성하기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))
#======================================================================
#3. 모델 실행 설정
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#======================================================================
#4. 모델 실행 설정
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
#======================================================================
#5. 레이블 준비
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#======================================================================
#6. 모델 훈련
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)
#======================================================================
#7. 모델 검증
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = Sequential()
model.add(Dense(512, activation='relu',input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc: ',test_acc)