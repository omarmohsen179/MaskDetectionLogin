import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import easygui

class App:
    def __init__(self, user,window, window_title,db, video_source=0):
        self.window = window
        self.db=db
        self.userid=user
        print(user)
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.window.geometry("820x580")
        self.window.resizable(width=False, height=False)
        self.canvas = tkinter.Canvas(window, width=640, height=480)
        self.canvas.pack()
        self.btn_snapshot = tkinter.Button(window, text="Login", width=50, command=self.snapshot,fg="white")
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        self.btn_snapshot['state'] = 'disabled'
        self.btn_snapshot['bg'] = 'red'
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1
        ap = argparse.ArgumentParser()
        ap.add_argument("-f", "--face", type=str,
                        default="face_detector",
                        help="path to face detector model directory")
        ap.add_argument("-m", "--model", type=str,
                        default="mask_detector.model",
                        help="path to trained face mask detector model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
        self.args = vars(ap.parse_args())

        # load our serialized face detector model from disk
        print("[INFO] loading face detector model...")
        prototxtPath = os.path.sep.join([self.args["face"], "deploy.prototxt"])
        weightsPath = os.path.sep.join([self.args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        # load the face mask detector model from disk
        print("[INFO] loading face mask detector model...")
        self.maskNet = load_model(
            self.args["model"]
        )
        print('camera-ON')

        try:
            self.c = cv2.VideoCapture(0)
            self.w, self.h = self.c.get(cv2.CAP_PROP_FRAME_WIDTH), self.c.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print('w:' + str(self.w) + 'px + h:' + str(self.h) + 'px')

        except:
            import sys
            print("error -----")
            self.c.release()
            cv2.destroyAllWindows()
        self.update()

        self.window.mainloop()

    def snapshot(self):
        mycursor = self.db.cursor()
        sql = "  UPDATE users SET Attendance = true WHERE id = "+str(self.userid)+" ; "
        mycursor.execute(sql)
        self.db.commit()
        self.window.destroy()
        easygui.msgbox("You have taked the attendace successfully", title="Mask Test")

        #easygui.msgbox("you have passed the test successfully", title="Mask Test")

    def detect_and_predict_mask(self, frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            if confidence > self.args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                except:
                    print("An exception occurred")

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            try:
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
            except:
                print("An exception occurred")

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            (locs, preds) = self.detect_and_predict_mask(frame, self.faceNet, self.maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                if label == "Mask":
                    color = (0, 255, 0)
                    self.btn_snapshot['state'] = 'normal'
                    self.btn_snapshot['bg'] = 'green'

                else:
                    color = (255, 0, 0)
                    self.btn_snapshot['state'] = 'disabled'
                    self.btn_snapshot['bg'] = 'red'
                # display the label and bounding box rectangle on the output
                # frame
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        ret, frame = self.vid.read()
        if self.vid.isOpened():
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
