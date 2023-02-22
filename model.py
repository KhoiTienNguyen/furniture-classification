from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Dense
from keras import backend as K
from keras import regularizers

import cv2
import requests
import numpy as np

class Model:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 2, input_shape=(None, None, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, 2))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3)))

        self.model.add(Conv2D(64, 3))
        self.model.add(Activation('relu'))

        self.model.add(GlobalMaxPooling2D())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3,kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
        self.model.add(Activation('softmax'))

        self.model.load_weights('best_model.hdf5')
        # self.model = load_model('best_model.hdf5')
        self.convert = {0: 'Bed', 1: 'Sofa', 2: 'Chair'}
    
    def predict(self, link):
        print(link)
        response =  requests.get(link).content
        nparr = np.frombuffer(response, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        scale_percent = 25 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_LANCZOS4)
        # print(self.model(np.expand_dims(resized, axis=(0,3))))
        # print()
        prediction = self.model(np.expand_dims(resized, axis=(0,3)))
        print(prediction)
        return self.convert[np.argmax(prediction)]
    
if __name__ == "__main__":
    model = Model()
    # result = model.predict('https://www.ikea.com/ca/en/images/products/hattefjaell-office-chair-with-armrests-smidig-black__1019087_pe831296_s5.jpg?f=xl')
    result = model.predict('google.com')
    
    print(result)