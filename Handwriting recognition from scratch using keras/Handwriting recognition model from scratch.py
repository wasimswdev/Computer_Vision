##importing the libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as k



##Training parameters
batch_size = 128
epochs = 5



##Loading the dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()


##reshaping training data from (60000,28,28) to (60000,28,28,1) as model expects
img_rows = x_train[0].shape[0]
img_col = x_train[1].shape[1]
x_train = np.reshape(x_train,(x_train.shape[0],img_rows,img_col,1))
x_test = np.reshape(x_test,(x_test.shape[0],img_rows,img_col,1))


##input shape
input_shape = (img_rows,img_col,1)


##converting to float
x_train = np.float32(x_train)
x_test = np.float32(x_test)


x_train /= 255
x_test /= 255


##one hot encoding the dependent variable 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


##getting no. of classes
num_classes = y_test.shape[1]



##building the  model
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))



##compiling the model
model.compile(optimizer=keras.optimizers.Adadelta(),loss='categorical_crossentropy',metrics=['accuracy'])



##model architecture
model.summary()

##training the model
model.fit(x_train,y_train,batch_size = batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test))




##testing accuracy
scores = model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


##Displaying the results(20 images)
def draw_text(name,pred,input_img):
    expanded_img = cv2.copyMakeBorder(input_img,0,0,0,imageL.shape[0],cv2.BORDER_CONSTANT,value=[0,0,0])
    expanded_img = cv2.cvtColor(expanded_img,cv2.COLOR_GRAY2RGB)
    cv2.putText(expanded_img,str(pred),(152,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),4)
    cv2.imshow(name,expanded_img)

for i in range(0,20):
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]
    imageL = cv2.resize(input_im,None,fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
    input_img = input_im.reshape(1,28,28,1)
    
    
    ##get prediction
    res = str(model.predict_classes(input_img,1,verbose=0)[0])
    draw_text("Prediction",res,imageL)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()