import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("photo_2022-05-04_17-07-15.jpg")
img_grey = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

faces = face_cascade_db.detectMultiScale(img_grey, 1.1, 19)
print(f"{len(faces)} лиц обнаружено на изображении.")
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)


cv2.imshow("result", img)
cv2.waitKey()