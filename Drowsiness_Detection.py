from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):
    playsound.playsound(path)  # chay am thanh tu duong dan

# ham tinh toan ty le co cua mat
def eye_aspect_ratio(eye):
    # tinh toan khoang cach euclid giua 2 tap hop
    # diem moc mat thang dung toa do (x,y)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # tinh toan khoang cach euclid giua phuong ngang
    # moc mat toa do (x,y)
    C = dist.euclidean(eye[0], eye[3])

    # tien hanh tinh toan
    ear = (A+B) / (2.0*C)

    return ear


ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")  # --shape-predictor: duong dan toi dlib de huan luyen facial landmark detector
# --alarm: tuy chinh am thanh dau vao nhu mot alarm
ap.add_argument("-a", "--alarm", type=str,
                default="./alarm.WAV", help="path alarm .WAV file")
# --webcam: so nguyen dieu khien gia tri duoc xay dung boi webcam
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

COUNTER = 0  # bien dem khi do gian no doi mat nho hon EYE_AR_THRESH
ALARM_ON = False  # mac dinh alarm khong duoc kich hoat

# khoi tao facial landmark detector
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# lay chi so tren khuon mat cua mat trai va mat phai
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def onStream(EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):
    # bat dau chay luong video truc tiep
    print("[INFO] starting video stream thread...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)

    # cho vong lap de lay hinh anh
    while True:
        global COUNTER
        # lay khung hinh tu luong video truc tiep, thay doi kich thuoc va chuyen doi thanh mau xam
        frame = vs.read()
        frame = imutils.resize(frame, width=480)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # phat hien guong mat tren khung xam
        rects = detector(gray, 0)

        maxBoundingBox = -1
        maxId = -1
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            total = (w+h)/2
            if total > maxBoundingBox:
                maxBoundingBox = total
                maxId = i
        # print(maxId)

        for (i, rect) in enumerate(rects):
            if i == maxId:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # trich xuat toa do cua mat trai va mat phai, tu do su dung cac toa do nay de tinh toan do gian no cua mat
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

            # tinh trung binh do gian no cua doi mat
                ear = (leftEAR + rightEAR) / 2.0

            # tinh do loi lom cua mat trai va mat phai, sau do truc quan hoa tung mat
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # neu do gian no mat duoi vung xac dinh  thi bien dem bat dau tang
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                # neu mat nham den so luong frame nhat dinh thi alarm duoc kich hoat
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True

                            if args["alarm"] != "":
                                t = Thread(target=sound_alarm,
                                           args=(args["alarm"],))
                                t.daemon = True
                                t.start()

                    # hien thi noi dung canh bao
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    COUNTER = 0
                    ALARM_ON = False

            # hien thi do gian no cua mat theo tg thuc
                cv2.putText(frame, "EAR: {:.2f}".format(
                    ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # hien thi hinh anh
        cv2.imshow("Driver Drowsiness Detection Video Stream", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit with q key
        if key == ord("q"):
            break

    # clean up after quit
    cv2.destroyAllWindows()
    vs.stop()


def onVideo(path, EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES):
    global COUNTER
    # bat dau chay luong video truc tiep
    print("[INFO] starting video thread...")
    vs = cv2.VideoCapture(path)
    time.sleep(1.0)

    # cho vong lap de lay hinh anh
    while (vs.isOpened()):
        # lay khung hinh tu luong video truc tiep, thay doi kich thuoc va chuyen doi thanh mau xam
        ret, frame = vs.read()
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # phat hien guong mat tren khung xam
        rects = detector(gray, 0)

        maxBoundingBox = -1
        maxId = -1
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            total = (w+h)/2
            if total > maxBoundingBox:
                maxBoundingBox = total
                maxId = i
        # print(maxId)

        for (i, rect) in enumerate(rects):
            if i == maxId:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # trich xuat toa do cua mat trai va mat phai, tu do su dung cac toa do nay de tinh toan do gian no cua mat
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

            # tinh trung binh do gian no cua doi mat
                ear = (leftEAR + rightEAR) / 2.0

            # tinh do loi lom cua mat trai va mat phai, sau do truc quan hoa tung mat
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # neu do gian no mat duoi vung xac dinh  thi bien dem bat dau tang
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                # neu mat nham den so luong frame nhat dinh thi alarm duoc kich hoat
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True

                            if args["alarm"] != "":
                                t = Thread(target=sound_alarm,
                                           args=(args["alarm"],))
                                t.daemon = True
                                t.start()

                    # hien thi noi dung canh bao
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    COUNTER = 0
                    ALARM_ON = False

            # hien thi do gian no cua mat theo tg thuc
                cv2.putText(frame, "EAR: {:.2f}".format(
                    ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # hien thi hinh anh
        cv2.imshow("Driver Drowsiness Detection Video File", frame)
        key = cv2.waitKey(1) & 0xFF

        # quit with q key
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
