import cv2
import numpy as np
import matplotlib.pyplot as plt
# Reading JPG Files
Nadia = cv2.imread("Nadia_Murad.jpg")
Denis = cv2.imread("Denis_Mukwege.jpg")
Solvay = cv2.imread("solvay_conference.jpg")

##### Defining xml files for features to detect the face #####

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#### Function to display images in matplotlib with color correction####

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    rgb_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_image)
    plt.show()

#### Function to detect faces in an image Provided ####

def detect_face(img):

    img_copy = img.copy()

    #roi = img.copy()

    face_rects = face_cascade.detectMultiScale(img_copy,scaleFactor=1.1,minNeighbors=5)

    # Grabing the points to draw the rectangle around the face
    for (x,y,w,h) in face_rects:

        #roi = roi[y:y+h,x:x+w]

        # better approach for streaming video(numpy slicing)
        img_copy[y:y+h,x:x+w] = cv2.medianBlur(img_copy[y:y+h,x:x+w],57)

        #img_copy[y:y+h,x:x+w] = blurred_roi

        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),3)

    return img_copy
#### for showing images ####
face = detect_face(Nadia)
# display(face)
#### Connection live camera to detect face ####
cap = cv2.VideoCapture(0)
while True:

    ret,frame = cap.read(0)

    if ret == True:

        frame = detect_face(frame)

        cv2.imshow("Face Detection",frame)

        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()