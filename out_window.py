
import os
import sys

from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QDate

import cv2

import numpy as np
import pickle
import time
import datetime
import imutils


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./untitled.ui", self)

#         Date and Time
        now__ = QDate.currentDate()
        current_date = now__.toString("ddd dd MMMM yyyy")
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.datevalue.setText(current_date)
        self.timevalue.setText(current_time)
        self.setWindowIcon(QIcon('icon.png'))

        self.livecam.clicked.connect(self.onClicked)
        self.browse.clicked.connect(self.browse_image)
    @pyqtSlot()
    def onClicked(self):

        curr_path = os.getcwd()
        print("Loading face detection model")
        proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
        model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
        face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

        print("loading recognition model")
        recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
        # the NN model is already available in opencv, let's retrieve it
        face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

        recognizer = pickle.loads(open('recognizer.pickle', 'rb').read())
        le = pickle.loads(open("le.pickle", 'rb').read())

        print("Starting test video file")
        vs = cv2.VideoCapture(0)
        time.sleep(1)  # delay execution

        while True:

            ret, frame = vs.read()
            frame = imutils.resize(frame, width=600)

            (h, w) = frame.shape[:2]

            image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                               False, False)

            face_detector.setInput(image_blob)
            face_detections = face_detector.forward()

            for i in range(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]

                if confidence >= 0.5:
                    box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]

                    (fH, fW) = face.shape[:2]

                    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

                    face_recognizer.setInput(face_blob)
                    vec = face_recognizer.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    info = name.split()
                    text = "{}: {:.2f}".format(name, proba * 100)
                    self.namevalue.setText("{}".format(info[0]))
                    self.agevalue.setText("{}".format(info[1]))
                    self.idvalue.setText("{}".format(info[2]))
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            self.displayImage(frame, 1)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

        cv2.destroyAllWindows()



    def displayImage(self, img, window=1):

        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.screenLabel.setPixmap(QPixmap.fromImage(img))
        self.screenLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Choose an image', 'C:/Users/pc/Images', 'JPG files (*.jpg)')
        # print(fname[0])
        self.screenLabel.setPixmap(QPixmap(fname[0]))
    def browse_image(self):
        filename = QFileDialog.getOpenFileName(self, 'Choose an image', 'C:/Users/pc/Images', 'JPG files (*.jpg)')
        curr_path = os.getcwd()
        print("Loading face detection model")
        proto_path = os.path.join(curr_path, 'model', 'deploy.prototxt')
        model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
        face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

        print("loading recognition model")
        recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
        # the NN model is already available in opencv, let's retrieve it
        face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

        recognizer = pickle.loads(open('recognizer.pickle', 'rb').read())
        le = pickle.loads(open("le.pickle", 'rb').read())

        print("Starting load image")

        # load the image

        image = cv2.imread(filename[0])


        # resisze our image
        frame = imutils.resize(image, width=353)


        (h, w) = frame.shape[:2]


        image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                           False, False)

        face_detector.setInput(image_blob)
        face_detections = face_detector.forward()


        for i in range(0, face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]

            if confidence >= 0.5:
                box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                (fH, fW) = face.shape[:2]

                face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

                face_recognizer.setInput(face_blob)
                vec = face_recognizer.forward()

                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                probaf = proba * 100
                text = "{}: {:.2f}".format(name, probaf)
                info = name.split()
                self.namevalue.setText("{}".format(info[0]))
                self.agevalue.setText("{}".format(info[1]))
                self.idvalue.setText("{}".format(info[2]))
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                print("accuracy level: ", probaf)

            cv2.imshow("frame", frame)
            self.displayImage(frame, 1)
            # key = cv2.waitKey(1) & 0xFF




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_OutputDialog()
    window.show()
    sys.exit(app.exec_())