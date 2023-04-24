# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:27:50 2023

@author: Dell
"""

import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2 as cv
from scipy import ndimage
################################################################################

#from random import Random
from tkinter import *
import tkinter
from tkinter.ttk import *
import time
import os
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import ttk
from PIL import Image
#import numpy as np
#from numpy import asarray
#Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image, ImageTk

#####################################################################################



Ram = Tk()
Ram.title("Naive Bayes Classification using MultiSpectral Remote Sensing Image")
Ram.minsize(1100, 1100)
Ram.maxsize(1100, 1100)
screen_width = Ram.winfo_screenwidth()
screen_height = Ram.winfo_screenheight()
winx = (screen_width/2)
winy = (screen_height/2)
"Ram.geometry('%dx%d+%d+%d' %(screen_width, screen_height, winx, winy))"
Ram.geometry('%dx%d+%d+4' %(screen_width, screen_height, winx))
Ram['background']='#124252'




Source = Text(Ram, height = 1, width = 44, bg = '#124252',relief=FLAT)


Heading = Text(Ram, height = 2, bg = '#0D2D37', relief=FLAT)
HeadingText='Naive Bayes Classification'
Heading.insert(tkinter.END,HeadingText)
Heading.pack()
Heading.config(font=("Times bold", 30))
Heading.config(fg="#E7A605")
Heading.config(state=DISABLED)
Hyperuploadtext=tkinter.StringVar()
Classyuuploadtext=tkinter.StringVar()


def open_train_file():
    global trainingData
    filetypes = (('TIFF image', '*.TIF'),('All files', '*.*'))
    trainingData = fd.askopenfilename(filetypes=filetypes)
    showinfo(title='Selected File', message=trainingData)
    Hyperuploadtext.set("Training Image uploaded")
def open_test_file():
    global testingData
    filetypes = (('TIFF image', '*.TIF'),('All files', '*.*'))
    testingData=fd.askopenfilename(filetypes=filetypes)
    showinfo(title='Selected File', message=testingData)
    Classyuuploadtext.set("Testing Image uploaded")

open_pan = ttk.Button(Ram, textvariable=Hyperuploadtext, command=open_train_file)
open_mss = ttk.Button(Ram, textvariable=Classyuuploadtext, command=open_test_file)
Hyperuploadtext.set("Upload Training image")
Classyuuploadtext.set("Upload Testing image")
open_pan.place(x=290, y=400)
open_mss.place(x=290, y=500)
style = Style()
style.configure('W.TButton',background = '#E7A605', foreground = '#124252', font='bold')



##################SANDHU#####################################
def fit(priors, means, vars, classes, image):
    priors = np.zeros(len(classes))
    means = np.zeros((len(classes), 4))
    vars = np.zeros((len(classes), 4))
    
    for i, c in enumerate(classes):
        a = image[:4, image[4] == c[2]]
        priors[i] = a.shape[1] / (image.shape[1] * image.shape[2])
        means[i, :] = np.mean(a, axis = 1)
        vars[i, :] = np.var(a, axis = 1)
    return priors, means, vars

def predict(priors, means, vars, classes, testImage):
    y_pred = np.zeros((3, testImage.shape[1], testImage.shape[2]))
    for j, k in np.ndindex((testImage.shape[1], testImage.shape[2])): 
        posteriors = np.zeros(len(classes))
        for i, c in enumerate(classes):
            prior = np.log(priors[i])
            likelihood = np.sum(normal_pdf(testImage[:4, j, k], means[i], vars[i]))
            posterior = likelihood
            posteriors[i] = prior + posterior
        y_pred[:, j, k] = classes[np.argmax(posteriors)]
    return y_pred

def normal_pdf(x, mean, var):
    coeff = np.log(1.0 / np.sqrt(2.0 * np.pi * var))
    exponent = -(x - mean)**2 / (2.0 * var)
    return coeff + exponent

report_label = Label(Ram, text="")

def gaussuian_blurr(image,kernel_size, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)
    normal = 1/(2* np.pi * sigma**2)
    gauss = np.exp(-((dst)**2 / (2.0 * sigma**2))) * normal
    blurred = cv.filter2D(src=image, ddepth=-1, kernel=gauss)
    return blurred

def Result():
    image = tif.imread(trainingData)
    testImage = tif.imread(testingData)
    benchmark = testImage[8:11]
    training_image = image[[1,2,4,6,10]]
    testing_image = testImage[[1,2,4,6,10]]

    training_image_smooth = np.zeros(training_image.shape)
    testing_image_smooth=np.zeros(testing_image.shape)

    kernel_size=9
    sigma=10

    for i in range(4):
        training_image_smooth[i]=gaussuian_blurr(training_image[i],kernel_size, sigma)
        testing_image_smooth[i]=gaussuian_blurr(testing_image[i],kernel_size, sigma)

    training_image_smooth[4] = training_image[4]
    testing_image_smooth[4] = testing_image[4]
    
    priors = None
    means = None
    vars = None
    classes1 = [[0,0,238],[38, 38, 38],[160, 30, 230], [255, 0, 0], [255, 255, 255], [238, 118, 33],[34, 139, 34],[0, 222, 137]]
    priors, means, vars = fit(priors, means, vars, classes1, training_image_smooth)

    test = np.zeros((3, image.shape[1], image.shape[2]))
    test = predict(priors, means, vars, classes1, testing_image_smooth)
    
    test = test.swapaxes(0, 2).swapaxes(0,1).astype(np.uint8)
    benchmark = benchmark.swapaxes(0,2).swapaxes(0,1).astype(np.uint8)

    report = f'Accuracy is {np.float16(accuracy_score(benchmark[:,:,2].flatten(), test[:,:,2].flatten())*100)}%'
    report_label.config(text=report, font='bold')

    test = Image.fromarray(test)
    benchmark = Image.fromarray(benchmark)

    benchmark_resized = benchmark.resize((685, 447), Image.ANTIALIAS)
    test_resized = test.resize((685, 447), Image.ANTIALIAS)

    global benchmark_image
    global test_image

    benchmark_image = benchmark_resized
    test_image = test_resized




def Output():
    win = Toplevel(Ram, height=screen_height, width= screen_width)
    win.attributes('-fullscreen', True)
    Text1=Label(win, text='Our Classification', font=('Bold', 44))
    Text1.place(x=100,y=12)
    Text2=Label(win, text='Benchmark', font=('Bold', 44))
    Text2.place(x=1000,y=12)

    classes_image = ImageTk.PhotoImage(Image.open('classes.jpg'))
    label3 = Label(win, image=classes_image)
    label3.image = classes_image
    label3.pack(side='bottom')

    test1 = ImageTk.PhotoImage(test_image)
    label1 = Label(win, image=test1, justify=CENTER)
    label1.image = test1
    label1.pack(side='left')

    test2 = ImageTk.PhotoImage(benchmark_image)
    label2 = Label(win, image=test2, justify=CENTER)
    label2.image = test2
    label2.pack(side='right')




Classify=ttk.Button(Ram,text="Classify",command=Result ,style = 'W.TButton')
Classify.place(x=290, y=600)

report_label.place(x=290, y=650)

Output=ttk.Button(Ram,text="Output",command=Output ,style = 'W.TButton')
Output.place(x=290, y=700)
Ram.mainloop()