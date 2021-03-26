# import playsound
import cv2
# from threading import Thread
import time
import proc

# Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier('./haar_cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haar_cascade/haarcascade_eye_tree_eyeglasses.xml')

wav_path = "./audio/alarm.wav"
detect_path = "./audio/detect.wav"
not_detect_path = "./audio/not_detect.wav"

# def sound_alarm(path):
#     playsound.playsound(path)

# Variable store execution state
ALARM_ON = False
COUNTER = 0
frames = 0
frames_not_Detect = 0

EYE_AR_CONSEC_FRAMES = 40


# Starting the video capture
cap = cv2.VideoCapture(1)
ret, img = cap.read()
time.sleep(0.1)

while (ret):
    ret, img = cap.read()
    # Coverting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray, 5, 1, 1)

    # Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    maxBoundingBox = -1
    maxId = -1

    if (len(faces) > 0):

        if frames == 1:
            if detect_path != "":
                proc.playSound(detect_path)
        for (i, rect) in enumerate(faces):
            (x, y, w, h) = rect
            region = (w + h)/2
            if region > maxBoundingBox:
                maxBoundingBox = region
                maxId = i

        for (i, rect) in enumerate(faces):
            if i == maxId:
                (x, y, w, h) = rect
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # roi_face is face which is input to eye classifier
                roi_face = gray[y:y + h, x:x + w]
                roi_face_clr = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_face, 1.3, 5, minSize=(40, 40))
                # print(eyes)
                # Examining the length of eyes object for eyes

                if (len(eyes) != None):
                    cv2.putText(img,"Eyes detected", (70, 70),cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 255), 2)
                    if(len(eyes) < 1):
                        COUNTER += 1

                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            if not ALARM_ON:
                                ALARM_ON = True

                                if wav_path != "":
                                    # t.start()
                                    proc.playSound(wav_path)

                            cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        COUNTER = 0
                        ALARM_ON = False
                    # print(COUNTER)

        frames += 1
        frames_not_Detect = 0
    else:
        if frames_not_Detect == 0:
            if detect_path != "":
                proc.playSound(not_detect_path)

        frames = 0
        frames_not_Detect += 1
        cv2.putText(img,
                    "No face detected", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 2)

    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if (a == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()