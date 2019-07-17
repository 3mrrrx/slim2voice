#python script to read 64 bytes of data from tiva C and plot them
#using pyQtGraph on a loop. As soon as 64 bytes arrive plot is updated

#import pyqtgraph ## Add path to library (just for examples; you do not need this)
from pyqtgraph import initExample

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

import cv2
import numpy as np
import time

# for logging function
import sys, os

# debug flags

show_imgs_for_colors = 1

def read_frame(scale_factor=1):

    # caputre frame
    ret, frame2 = cap.read()

    # rescale image for perfromance
    scale =  1 / len(colors) / scale_factor
    imgs = cv2.resize(frame2, (0, 0), None, scale, scale)

    return imgs, frame2


def percent_color(img, color_min = np.array([0, 0, 128], np.uint8), color_max= np.array([250, 250, 255], np.uint8) ):
    #RED_MIN = np.array([0, 0, 128], np.uint8)
    #RED_MAX = np.array([250, 250, 255], np.uint8)

    size = img.size

    dstr = cv2.inRange(img, color_min, color_max)
    no_color = cv2.countNonZero(dstr)
    frac_color = np.divide((float(no_color)), (int(size)))
    percent_color = np.multiply((float(frac_color)), 100)

    #print('color: ' + str(percent_color) + '%')

    return percent_color



def percent_color_singel(img,color = [145,80,40], threshold = 20, disp = "image" ):

    boundaries = [([max(color[2] - threshold,0),max(color[1] - threshold,0),max(color[0] - threshold,0)],
                   [min(color[2] + threshold,255), min(color[1] + threshold,255), min(color[0] + threshold,255)])]
    # in order BGR as opencv represents images as numpy arrays in reverse order

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)

        output = cv2.bitwise_and(img, img, mask=mask)

        ratio = cv2.countNonZero(mask) / (img.size / 3)
        perc = np.round(ratio * 100, 2)
        #print('the color: ' + str(color) + ' ,has a pixel percentage:', perc, '%.')

        ##cv2.imshow( disp, np.hstack([img, output]))
        #cv2.imshow(disp, output)

        #cv2.waitKey(0)

        return perc, output

def percent_color_singel_lab(img,color = [145,80,40], thresh = 20, disp = "image" ):
    #convert 1D array to 3D, then convert it to LAB and take the first element
    #lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]
    lab = color

    minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

    maskLAB = cv2.inRange(img, minLAB, maxLAB)
    output = cv2.bitwise_and(img, img, mask = maskLAB)

    ratio = cv2.countNonZero(maskLAB) / (img.size / 3)
    perc = np.round(ratio * 100, 2)

    return perc, output

def FFT_AMP(data):
        data= data - data.mean()
        data=np.hamming(len(data))*data
        data=np.fft.fft(data)
        data=np.abs(data)
        return data

def cvNpArrayRGB2LAB(array):
        array = np.array(array)
        color_lab = np.array(array, np.uint8)

        # print(np.iinfo(color.dtype))
        # info = np.iinfo(color.dtype) # Get the information of the incoming image type
        # color = color.astype(np.float64) / info.max # normalize the data to 0 - 1
        # color = 255 * color # Now scale by 255
        # color = color.astype(np.uint8)
        # print(color)
        color_lab = cv2.cvtColor(np.uint8([[color_lab]])  , cv2.COLOR_BGR2LAB)
        #colors_lab.append(color_lab)
        return color_lab[0][0]


###########################################################################################################################
# video source
###########################################################################################################################

# web cam
cap = cv2.VideoCapture(0)

###########################################################################################################################
# initale varaibles
###########################################################################################################################

# colors to select from

#colors = [([145, 80, 40], 50), ([50, 80, 255], 50), ([185, 130, 51], 50), ([50, 10, 39], 50), ([50, 80, 39], 50)]
#colors = [([255, 0, 0], 50), ([0, 0, 0], 50), ([135,4,0], 50), ([0, 255, 0], 50), ([200, 130, 180], 50),([255, 0, 255], 50),([185,124,109], 50)]
#colors = [([150, 150, 150], 120), ([0, 0, 0], 50), ([150, 150, 150], 70), ([0, 0, 0], 20)]
#colors = [ ([150, 150, 150], 70), ([0, 0, 0], 20)]

colors = [([135,4,0], 50), ([0,0,0], 30)]

# convert RGB to lab color space
colors_lab = []
for color, th in colors:
    color_lab = cvNpArrayRGB2LAB(color)

    colors_lab.append((color_lab,th))

colors_lab = [([135,130,240], 50), ([0,0,0], 30)]

# number of saved frames for witch the perc. are saved
perc_buffer = 128

#color_perc_list = [] # list of all color percentages of all frames
color_perc_list = np.empty([len(colors),perc_buffer])
color_perc_list_lab = np.empty([len(colors_lab),perc_buffer])

###########################################################################################################################
# capture first frame
###########################################################################################################################

# start fps timer
t = time.time()
frame_num = 0

# capture first frame
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

###########################################################################################################################
# create plot window
###########################################################################################################################

app = QtGui.QApplication([])

# set background color to white
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# create window for graphs
win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

#win = pg.GraphicsWindow()

###########################################################################################################################
# show frames  in qt
###########################################################################################################################

# pimg = win.addPlot(title="imgs")
#
# pq_images = []
# for color ,threshold in colors:
#    pq_images.append(pimg.ImageView())
#
# imv = pg.ImageView()
# imv.show()
#
# win.nextRow()


###########################################################################################################################
# color plot
###########################################################################################################################

p = win.addPlot(title="Color Progration Over Time")

p.plot()
p.setWindowTitle('colors')
p.setLabel('bottom', 'Index', units='B')
p.showGrid(x=True, y=True, alpha=None)
#p.setLimits(yMin=0,yMax=100)
#p.enableAutoRange(axis=None, enable=False)
p.addLegend()

lines_prog = []
for color ,threshold in colors:
    #color_scaled = [x for x in color]
    rgb_color = tuple(color)
    lines_prog.append(p.plot(pen=rgb_color, width=10, name=str(color),row=0, col=0))

###########################################################################################################################
# color plot
###########################################################################################################################

p_lab = win.addPlot(title="Color Progration Over Time")

p_lab.plot()
p_lab.setWindowTitle('colors lab')
p_lab.setLabel('bottom', 'Index', units='B')
p_lab.showGrid(x=True, y=True, alpha=None)
#p.setLimits(yMin=0,yMax=100)
#p.enableAutoRange(axis=None, enable=False)
p_lab.addLegend()

lines_prog_lab = []
for color ,threshold in colors_lab:
    lines_prog_lab.append(p_lab.plot(pen=rgb_color, width=10, name=str(color_lab),row=0, col=0))

win.nextRow()


###########################################################################################################################
# RGB Histogram plot
###########################################################################################################################

p_hist = win.addPlot(title="Color Histogram")
p_hist.plot()
p_hist.setWindowTitle('histogram')
p_hist.setLabel('bottom', 'Index', units='B')
p_hist.addLegend()
p_hist.showGrid(x=True, y=True, alpha=None)
p_hist.setLimits(yMax=1,yMin=0,xMin=0)

phist_colors = ('b', 'g', 'r') # cv2 convention of rgb
phist = []
for color in phist_colors:
    rgb_color = tuple(color)
    phist.append(p_hist.plot(pen=color, width=10, name=str(color)))


###########################################################################################################################
# histogram LAB color space
###########################################################################################################################


p_hist_lab = win.addPlot(title="Color Histogram LAB ")
p_hist_lab.plot()
p_hist_lab.setWindowTitle('histogram lab')
p_hist_lab.setLabel('bottom', 'Index', units='B')
p_hist_lab.addLegend()
p_hist_lab.showGrid(x=True, y=True, alpha=None)
p_hist_lab.setLimits(yMax=1,yMin=0,xMin=0)

phist_colors_names = ('y', 'k', 'r')

phist_lab = []
name_lab = ["l","a","b"]
for i, color in enumerate(name_lab):
    rgb_color = tuple(color)
    phist_lab.append(p_hist_lab.plot(pen=phist_colors_names[i], width=10, name=str(name_lab[i])))

win.nextRow()


###########################################################################################################################
# fft plot
###########################################################################################################################

RATE = 30
data_fft=np.zeros(perc_buffer)
axis_fft=np.fft.fftfreq(perc_buffer, d=1.0/RATE)

p_fft = win.addPlot(title="FFT")
p_fft.plot()
p_fft.setWindowTitle('FFT')
p_fft.setLabel('bottom', 'Index', units='B')
p_fft.addLegend()
p_fft.showGrid(x=True, y=True, alpha=None)
p_fft.setLimits(yMax=100,yMin=0)
#p_fft.autoRange(enable=False)

pfft = []
for color ,threshold in colors:
    #color_scaled = [x for x in color]
    rgb_color = tuple(color)
    pfft.append(p_fft.plot(axis_fft,data_fft, pen=rgb_color, width=10, name=str(color),row=0, col=1))

###########################################################################################################################
# fft lab plot
###########################################################################################################################

RATE_lab = 30
data_fft_lab=np.zeros(perc_buffer)
axis_fft_lab=np.fft.fftfreq(perc_buffer, d=1.0/RATE)

p_fft_lab = win.addPlot(title="FFT LAB")
p_fft_lab.plot()
p_fft_lab.setWindowTitle('FFT')
p_fft_lab.setLabel('bottom', 'Index', units='B')
p_fft_lab.addLegend()
p_fft_lab.showGrid(x=True, y=True, alpha=None)
p_fft_lab.setLimits(yMax=100,yMin=0)
#p_fft.autoRange(enable=False)

pfft_lab = []

index = 0
for color_lab ,threshold in colors_lab:

        # color_ab = np.array(color, np.uint8)
        #
        # # print(np.iinfo(color.dtype))
        # # info = np.iinfo(color.dtype) # Get the information of the incoming image type
        # # color = color.astype(np.float64) / info.max # normalize the data to 0 - 1
        # # color = 255 * color # Now scale by 255
        # # color = color.astype(np.uint8)
        # # print(color)
        # color_lab = cv2.cvtColor( np.uint8([[color_ab]])  , cv2.COLOR_BGR2LAB)
        # colors_lab.append(color_lab)

    rgb_color = tuple(colors[index][0])
    pfft_lab.append(p_fft_lab.plot(axis_fft_lab,data_fft_lab,\
        pen=rgb_color, width=10, name=str(colors_lab),row=0, col=1))

    index += 1


###########################################################################################################################
# update
###########################################################################################################################


def update():
    global lines_prog, data, t, frame_num,\
     colors, colors_lab, color_perc_list, color_perc_list_lab, \
     show_imgs_for_colors, p, pfft, data_fft, axis_fft, RATE, phist_colors

    ######################################################################
    # read frame
    ######################################################################
    img, frame2 = read_frame(0.5)

    # update lab colorspace histogram and set data for plot
    frame2Lab = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    ######################################################################
    # calatulate color percentages
    ######################################################################
    color_prec_frame = [] # list of color percentages for current frame
    imgs = img # save orginal image
    index = 0

    for color, thresh in colors:
        color_pers_i, img = percent_color_singel(img,\
         color=color, threshold=thresh, disp= str(color))
        color_prec_frame.append(color_pers_i)
        imgs = np.hstack((imgs,img))
        index += 1

    if show_imgs_for_colors ==  0:
       #images_per_row = 2
       #cv2.imshow( "ALL_1", np.hstack(imgs))
       cv2.imshow( "ALL_1", imgs)

    # add color from frame the last frames perc list
    color_perc_list = np.roll(color_perc_list, -1, axis=1)
    color_perc_list[:,-1] = color_prec_frame


    ######################################################################
    # calatulate lab color percentages
    ######################################################################
    color_prec_frame_lab = [] # list of color percentages for current frame
    imgs_lab = img_lab # save orginal image
    index = 0

    for color, thresh in colors_lab:

        color_pers_i_lab, img_lab = percent_color_singel_lab(img_lab,\
         color=color, thresh=thresh, disp= str(color))
        color_prec_frame_lab.append(color_pers_i_lab)
        imgs_lab = np.hstack((imgs_lab,img_lab))
        index += 1

    if show_imgs_for_colors ==  1:
       #images_per_row = 2
       #cv2.imshow( "ALL_1", np.hstack(imgs))
       cv2.imshow( "ALL_1", imgs_lab)

    # add color from frame the last frames perc list
    color_perc_list_lab = np.roll(color_perc_list_lab, -1, axis=1)
    color_perc_list_lab[:,-1] = color_prec_frame_lab

    ######################################################################
    # plots
    ######################################################################

    # update data for line Progration
    for i, x in enumerate(lines_prog):
        #print(color_perc_list[i,:])
        x.setData(color_perc_list[i,:])

    # update data for line Progration in lab space
    for i, x in enumerate(lines_prog_lab):
        #print(color_perc_list[i,:])
        x.setData(color_perc_list_lab[i,:])

    ## change foor loop to map
    # map(lambda x,y: x.setData(y), lines_prog, color_perc_list.tolist())

    # update RGB color space histogram and set plot data
    for i, x in enumerate(phist):
        histr = cv2.calcHist([frame2], [i], None, [256], [0, 256])
        histr = cv2.normalize(histr,histr)
        #print(np.shape(histr))
        x.setData(np.reshape(histr, np.shape(histr)[0]))

    # update fft and set data for plot
    for i, x in enumerate(pfft):

        # calc fft
        data_fft = color_perc_list[i,:]
        fft_data=FFT_AMP(data_fft)
        axis_pfft=np.fft.fftfreq(len(data_fft), d=1.0/RATE)

        #plot data
        x.setData(x=np.abs(axis_fft), y=fft_data)

    for i, x in enumerate(phist_lab):
        histr = cv2.calcHist([frame2Lab], [i], None, [256], [0, 256])
        histr = cv2.normalize(histr,histr)
        #print(np.shape(histr))
        x.setData(np.reshape(histr, np.shape(histr)[0]))

    # update fft lab and set data for plot
    for i, x in enumerate(pfft_lab):

        # calc fft lab
        data_fft_lab = color_perc_list[i,:]
        fft_data_lab=FFT_AMP(data_fft_lab)
        axis_pfft_lab=np.fft.fftfreq(len(data_fft_lab), d=1.0/RATE_lab)

        #plot data
        x.setData(x=np.abs(axis_pfft_lab), y=fft_data_lab)


    #  calc frame rate
    if frame_num%10 == 0:
        elapsed = time.time() - t
        print("fps: " + str(10/elapsed))
        t = time.time()

    app.processEvents()  ## force complete redraw for every plot




timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

###########################################################################################################################
# main
###########################################################################################################################

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
