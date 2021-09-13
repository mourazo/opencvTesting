import cv2
import os
import imutils
import sys

sys.path.append('c:/users/user/appdata/local/programs/python/python39/lib/site-packages')#this is for the case you recive cv2 errors

personName = 'personName' #set the name of the person for the training model
dataPath = 'D:/python/code/ia/facialRecognition/data' #your route for the output images
personPath = dataPath + '/' + personName #set the name for the folder

if not os.path.exists(personPath): #create the folder
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #open the cam
#cap = cv2.VideoCapture('Video.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #make a cascade classifier object detection
count = 0

while True:
    
    ret, frame = cap.read() #read the cam
    if ret == False: break #if the cam not present close
    frame =  imutils.resize(frame, width=640) #function to mantain the aspect ratio
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #space color for the frame
    auxFrame = frame.copy() 

    faces = faceClassif.detectMultiScale(gray,1.3,5)#detect the faces

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #set the rectangle in the faces 
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC) #change the dimensions according the distances
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro) #save the images of the faces
        count = count + 1
    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()