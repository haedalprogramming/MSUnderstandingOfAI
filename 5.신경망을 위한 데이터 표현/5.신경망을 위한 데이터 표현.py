#======================================================================
# 5.2 넘파이로 텐서 조작하기
#1. MINIST 데이터넷 불러오기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()
#======================================================================
#2. train_images 크기를 확인하기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()

train_images.shape
#======================================================================
#3. 특정 샘픔을 선택해보기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()

my_slice = train_images[10:100]
my_slice.shape
#======================================================================
#4. 이미지 슬라이싱 해보기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(train_images[4], cmap=plt.cm.binary)
plt.show()
#======================================================================
#5. width와 height를 슬라이싱 해보기
#======================================================================
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels)=mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(train_images[4, 5:-5,5:-5], cmap=plt.cm.binary)
plt.show()
#======================================================================
# 5.4 텐서의 크기 변환
#======================================================================
import numpy as np

x = np.array([1,2,3,4,5,6])
print(x.shape)

y = x.reshape((2,3))
print(y.shape)

z = x.reshape(-1,1)
print(z)
print(z.shape)

z = x.reshape(1,-1)
print(z)
print(z.shape)


