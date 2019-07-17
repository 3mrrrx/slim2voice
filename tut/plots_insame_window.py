#python script to read 64 bytes of data from tiva C and plot them
#using pyQtGraph on a loop. As soon as 64 bytes arrive plot is updated

import pyqtgraph/initExample ## Add path to library (just for examples; you do not need this)


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

        ratio_brown = cv2.countNonZero(mask) / (img.size / 3)
        perc = np.round(ratio_brown * 100, 2)
        #print('the color: ' + str(color) + ' ,has a pixel percentage:', perc, '%.')

        ##cv2.imshow( disp, np.hstack([img, output]))
        #cv2.imshow(disp, output)

        #cv2.waitKey(0)

        return perc, output

def FFT_AMP(data):
        data= data - data.mean()
        data=np.hamming(len(data))*data
        data=np.fft.fft(data)
        data=np.abs(data)
        return data


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

# number of saved frames for witch the perc. are saved
perc_buffer = 128

#color_perc_list = [] # list of all color percentages of all frames
color_perc_list = np.empty([len(colors),perc_buffer])
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
# pg.setConfigOption('background', 'w')
# pg.setConfigOption('foreground', 'k')

# create window for graphs
win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

#win = pg.GraphicsWindow()

###########################################################################################################################
# print frames
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
p.setWindowTitle('FFT')
p.setLabel('bottom', 'Index', units='B')
p.showGrid(x=True, y=True, alpha=None)
#p.setLimits(yMin=0,yMax=100)
#p.enableAutoRange(axis=None, enable=False)
p.addLegend()

lines_prog = []
for color ,threshold in colors:
    #color_scaled = [x for x in color]
    rgb_color = tuple(color)
    #lines_prog.append(ax_prog.plot([], [], color=rgb_color, label=str(color)))
    #lines_prog.append(p.plot( pen=pg.mkPen(pg.QColor(rgb_color), width=5), name=str(rgb_color)))
    #lines_prog.append(p.plot( pen=QPen(QColor(255,0,0))))
    lines_prog.append(p.plot(pen=rgb_color, width=10, name=str(color),row=0, col=0))
    #lines_prog.append(win.addPlot(pen=rgb_color, width=10, name=str(color_scaled),row=0, col=0))

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
# fft plot
###########################################################################################################################
win.nextRow()

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
    color_scaled = [x for x in color]
    rgb_color = tuple(color_scaled)
    pfft.append(p_fft.plot(axis_fft,data_fft, pen=rgb_color, width=10, name=str(color_scaled),row=0, col=1))


###########################################################################################################################
# histogram LAB color space
###########################################################################################################################


p_hist_lab = win.addPlot(title="Color Histogram LAB ")
p_hist_lab.plot()
p_hist_lab.setWindowTitle('histogram')
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


###########################################################################################################################
# update
###########################################################################################################################


def update():
    global lines_prog, data, t, frame_num, colors, color_perc_list, \
     show_imgs_for_colors, p, pfft, data_fft, axis_fft, RATE, phist_colors

    # read frame
    img, frame2 = read_frame(0.5)

    # calatulate color percentage
    color_prec_frame = [] # list of color percentages for current frame
    imgs = img # save orginal image
    index = 0
    for color, threshold in colors:
        color_pers_i, img = percent_color_singel(img, color=color, threshold=threshold, disp= str(color))
        color_prec_frame.append(color_pers_i)
        imgs = np.hstack((imgs,img))
        index += 1

    if show_imgs_for_colors ==  1:
       #images_per_row = 2
       #cv2.imshow( "ALL_1", np.hstack(imgs))
       cv2.imshow( "ALL_1", imgs)

    # for x in enumerate(pq_images):
    #     #pimg.image(x)
    #     x.setImage(imgs[i])

        #x.setData(imgs[i])

    # add color from frame the last frames perc list
    color_perc_list = np.roll(color_perc_list, -1, axis=1)
    color_perc_list[:,-1] = color_prec_frame

    # update data for line Progration
    for i, x in enumerate(lines_prog):
        #print(color_perc_list[i,:])
        x.setData(color_perc_list[i,:])

#    map(lambda x,y: x.setData(y), lines_prog, color_perc_list.tolist())

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

    # update lab colorspace histogram and set data for plot
    frame2Lab = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)
    for i, x in enumerate(phist_lab):
        histr = cv2.calcHist([frame2Lab], [i], None, [256], [0, 256])
        histr = cv2.normalize(histr,histr)
        #print(np.shape(histr))
        x.setData(np.reshape(histr, np.shape(histr)[0]))


    #  calc frame rate
    if frame_num%10 == 0:
        elapsed = time.time() - t
        print("fps: " + str(10/elapsed))
        t = time.time()

    app.processEvents()  ## force complete redraw for every plot




timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
