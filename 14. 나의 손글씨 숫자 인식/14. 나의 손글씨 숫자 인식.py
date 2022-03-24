#======================================================================
#14.2 모델 로드
#가. 모델 파일을 로드
#======================================================================
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')
#======================================================================
#나. 모델 요약 정보를 확인
#======================================================================
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')

model.summary()
#======================================================================
#14.3 이미지 확인 및 변환
#======================================================================
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('5.png')
plt.imshow(img);

img.size

img = np.asarray(img)
#======================================================================
#14.4 이미지 인식
#======================================================================
from tensorflow.keras.models import load_model

model = load_model('13-0.0257.hdf5')

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open('5.png')
plt.imshow(img);

img.size

img = np.asarray(img)

res = model.predict(np.reshape(img, (1, 28, 28, 1)))
res

np.argmax(res[0])