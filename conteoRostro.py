import cv2
import os
import numpy as np
dataPath = 'D:/python/code/ia/facialRecognition/data'#your path
peopleList = os.listdir(dataPath) #list the files of the folder
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0
for nameDir in peopleList: #reading the names of the images
    personPath = dataPath + '/' + nameDir 
    print('Leyendo las imágenes')
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0)) #take the images of the file
        #image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label = label + 1
#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create() #create the LBPH model
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels)) #training the model
face_recognizer.write('modeloLBPHFace.xml') #write the model
print("Modelo almacenado...")