import cv2
import os
import face_recognition

dataPath = 'D:/python/code/ia' #path where is the file xml EigenFaces, FisherFace o LBPHFace
imagePaths = os.listdir(dataPath) #list of the files in the path
print('imagePaths=',imagePaths) 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create the face_recognizer LBPH
# reading the model
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPHFace.xml') #path for read the face_recognizer
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #open your cam for the video input
#cap = cv2.VideoCapture('Video.mp4')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #cascade clasifier
while True:
    ret,frame = cap.read() #read the video data
    if ret == False: break #if no video, close
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #space color for the frame
    auxFrame = gray.copy() #using the COLOR_BGR2GRAY
    faces = faceClassif.detectMultiScale(gray,1.3,5) #scale for the faces
    for (x,y,w,h) in faces: #method for recognize the face
        rostro = auxFrame[y:y+h,x:x+w] #set the auxframe surrounding the face
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC) #changing the dimensions of the rectangle
        result = face_recognizer.predict(rostro) #make the face recognizer into the auxframe
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        '''
        # EigenFaces
        if result[1] < 5700:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        
        # FisherFace
        if result[1] < 500:
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        '''
        #LBPHFace
        if result[1] < 70: #if we have concidences with our xml
            cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA) #display the name of the training model
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2) #display a rectangle surrounding
        else:
            cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA) #display the unknowledge label
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2) #display a red rectangle surrounding
        
    cv2.imshow('frame',frame) #displaying the image
    k = cv2.waitKey(1)  #wait a key for exit
    if k == 27: #if key is escape, close
        break
cap.release()
cv2.destroyAllWindows()