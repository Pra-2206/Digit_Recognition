import cv2
import numpy as np
import tensorflow as tf
m_new = tf.keras.models.load_model('/home/prashant/Downloads/model_mnist.h5')
draw = False
img = np.ones([300,300])*255
windowname = 'Digit_Recognition'
cv2.namedWindow(windowname)
def demo(event,x,y,flags,param):
    global draw
    if event==cv2.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            cv2.circle(img,(x,y),4,(0,0,0),5)
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
cv2.setMouseCallback(windowname,demo)
while(True):
    cv2.imshow(windowname,img)
    if cv2.waitKey(1) == ord('c'):
        img[:,:] = 255
        
    elif cv2.waitKey(1) == ord('p'):
        out = img[:,:]
        out1 = cv2.resize(out,(28,28)).reshape(1,28,28,-1)
        print(m_new.predict_classes(out1))
        
    elif cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
    
