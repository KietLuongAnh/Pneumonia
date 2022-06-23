from dataclasses import field
import sys
import os
import cv2
from cv2 import IMREAD_COLOR
from cv2 import FONT_HERSHEY_COMPLEX
from cv2 import FONT_HERSHEY_SIMPLEX
import numpy as np
import tkinter
from tkinter import *
from tkinter.filedialog import *
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
global pred
pred = 2
detection_model = load_model('xinxo.h5')
diag = ["Chuẩn đoán bình thường", "Chuẩn đoán mắc viêm phổi","Hello"]
window = Tk()
window.title('Kết quả chuẩn đoán')
window.geometry("320x120+100+100")
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
class Main(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.InitUI()
    
    def InitUI(self):
        self.parent.title("Tool Chuẩn đoán bệnh Viêm Phổi")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label= "Open", command=self.onOpen)
        fileMenu.add_command(label= "Diagnostic", command=self.OnRecognition)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label= "File", menu= fileMenu)
        self.txt = Text(self)
        self.txt.pack(fill=BOTH, expand=1)

    def onOpen(self):
        global ftypes 
        ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png *.jpeg')]
        dlg = Open(self, filetypes = ftypes)
        fl = dlg.show()

        if fl != '':
            global imgin 
            cv2.destroyAllWindows()
            imgin = cv2.imread(fl, cv2.IMREAD_COLOR)
            imgin = cv2.resize(imgin,(450,450))
            # print(fl)
            cv2.namedWindow("ImageIn", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("ImageIn", imgin)
            # plt.imshow(imgin)
            # plt.show()

            

    def OnRecognition(self):
        global show
        image = img_to_array(imgin)
        image = cv2.resize(imgin, (150, 150))
        image = image.reshape(-1, 150, 150, 3)
        image = image.astype('float32')
        image /= 255

        pred = np.argmax(detection_model.predict(image))
        print(type(pred))
        print(pred)
        lbl = Label(window, text = diag[pred])
        lbl.grid(column=0,row=0)

        if pred == 0:
            print("Chuẩn đoán bình thường")
        elif pred == 1:
            print("Chuẩn đoán viêm phổi")

root = Tk()
Main(root)
root.geometry("320x120+100+100")
root.mainloop()