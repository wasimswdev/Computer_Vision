import cv2
import numpy as np
import keras
from keras.models import load_model
from keras.datasets import mnist



##loading the pre-trained saved model
model = load_model('mnist_simple_cnn.h5')



##loading the dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()




##displaying the result
def draw_text(name,pred,input_img):
    expanded_img = cv2.copyMakeBorder(input_img,0,0,0,imageL.shape[0],cv2.BORDER_CONSTANT,value=[0,0,0])
    cv2.putText(expanded_image,str(pred),(152,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,4,(0,255,0),2)
    cv2.imshow(name,expanded_image)

for i in range(0,10):
    rand = np.random.randint(0,len(x_test))
    input_im = x_test[rand]
    imageL = cv2.resize(input_im,None,fx=4,fy=4,interpolation = cv2.INTER_CUBIC)
    input_img = input_im.reshape(1,28,28,1)
    
    
    ##get prediction
    res = str(model.predict_classes(input_img,1,verbose=0)[0])
    draw_text("Prediction",res,imageL)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()