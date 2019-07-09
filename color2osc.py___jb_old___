
from matplotlib import pyplot as plt, style
import cv2
import numpy as np
import time
from datetime import datetime

# to send osc messages
import argparse
from pythonosc import udp_client

# for logging function
import sys
import os

# matplotlib color scheam
plt.style.use('fivethirtyeight')

###########################################################################################################################
# debug
###########################################################################################################################

histogram = 0           # plots a live histogram of the colors
show_org = 0            # shows a live image from orginal video
motion_dictection = 0   # not implanted
save_image = 0          # saves the recorded video to creat loops (not full implanted)
save_perc_graph = 0     # saves the finale version of the recorded histogram of the colors
print_color_prog = 1    #

###########################################################################################################################
# defs
###########################################################################################################################
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

        # write image to file with time as name
        if save_image == 1:
            millis = int(round(time.time() * 1000))
            cv2.imwrite("pictures2\\" + disp + str(millis) + ".png", output)

        #cv2.waitKey(0)

        return perc, output


###########################################################################################################################
# osc functions
###########################################################################################################################


def rescale(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


# def for OSC messages
def send_freqs_to_synth_0(freqs):
    client.send_message("/filter_0", freqs)

def send_freqs_to_synth(freqs):
    client.send_message("/filter", freqs)

def send_freqs_to_synth_2(freqs):
    client.send_message("/filter_2", freqs)

def send_freqs_to_synth_3(freqs):
    client.send_message("/filter_3", freqs)

def send_freqs_to_synth_4(freqs):
    client.send_message("/filter_4", freqs)

def send_freqs_to_synth_5(freqs):
    client.send_message("/filter_5", freqs)

def send_freqs_to_synth_6(freqs):
    client.send_message("/filter_6", freqs)

# def for playing OSC messages from inputs
def play_synth(argument):

    # set up scale
    fundamental = 40 # fundamental note of scale
    a = 24 # change octave for player a
    b = 0  # change octave for player b

    # send OSC messages
    for i , x in enumerate(argument):

        x = rescale(float(x)/100,[0,1], [50,10000])
        client.send_message("/filter_"+str(i), str(x))


###########################################################################################################################
# video source
###########################################################################################################################

# web cam
cap = cv2.VideoCapture(0)

# or read video
# path = os.path.join("..", "video", )
#path = "C:\\Users\\hasan\\Desktop\\Holyspace\\kasia\\videos\\blood_maps.mov"
#path = "C:\\Users\\hasan\\Desktop\\Holyspace\\kasia\\videos\\blood_maps_small.mp4"
#cap = cv2.VideoCapture(path)

###########################################################################################################################
# initale varaibles
###########################################################################################################################




t = time.time()
frame_num = 0

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# colors to select from

#colors = [([145, 80, 40], 50), ([50, 80, 255], 50), ([185, 130, 51], 50), ([50, 10, 39], 50), ([50, 80, 39], 50)]
#colors = [([255, 0, 0], 50), ([0, 0, 0], 50), ([135,4,0], 50), ([0, 255, 0], 50), ([200, 130, 180], 50),([255, 0, 255], 50),([185,124,109], 50)]
colors = [([120, 0, 0], 100), ([0, 120, 0], 100), ([135,4,30], 100), ([0, 120, 0], 100), ([150, 0, 180], 100),([130, 0, 130], 100),([155,124,0], 100)]

color_pers_list = [] #np.empty([1, len(colors)])

###########################################################################################################################
# set up OSC
###########################################################################################################################

# check super collider for parser


# set up log file name

#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))

#f = open('output.txt', 'a+')
#f.write('Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "\n")
#f.close()

print("setting put OSC client")

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.178.93",
                    help="The ip of the OSC server")
parser.add_argument("--port", type=int, default=9000,
                    help="The port the OSC server is listening on")
args = parser.parse_args()

client = udp_client.SimpleUDPClient(args.ip, args.port)


###########################################################################################################################
# start loop
###########################################################################################################################
while(1):

    ###########################################################################################################################
    # read frame
    ###########################################################################################################################

    ret, frame2 = cap.read()


    ###########################################################################################################################
    # old methods
    ###########################################################################################################################

    #perce_red = percent_color(frame2,color_min = np.array([0, 0, 128], np.uint8), color_max= np.array([250, 250, 255], np.uint8))
    #perce_green =  percent_color(frame2,color_min = np.array([0, 0, 0], np.uint8), color_max= np.array([250, 250, 128], np.uint8))
    #perce_blue = percent_color(frame2,color_min = np.array([0, 0, 0], np.uint8), color_max= np.array([250, 250, 128], np.uint8) )

    #blue, green, red = cv2.split(frame2)

    #n_red = np.sum(red)
    #n_blue = np.sum(blue)
    #n_green = np.sum(green)

    #total = n_blue + n_red + n_green

    #print("red= " + str(n_red/total) + "% " + "green= " + str(n_green/total) + "% " + "blue= " + str(n_blue/total) + "% " )

    ###########################################################################################################################
    # motion dictection
    ###########################################################################################################################

    if motion_dictection == 1:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = 0#ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2',bgr)

    ####################################################################################################
    # calatulate color pescetage
    ####################################################################################################

    # rescale image for perfromance
    scale =  1 / len(colors)
    resize_frame2 = cv2.resize(frame2, (0, 0), None, scale, scale)

    i = 0
    color_prec_frame = [] # list with color persecnt for current frame
    imgs = resize_frame2

    for color, th in colors:
        color_pers_i, img = percent_color_singel(resize_frame2, color=color, threshold=th, disp= str(color))
        color_prec_frame.append(color_pers_i)
        #imgs.append(img)
        imgs = np.hstack((imgs,img))
    color_pers_list.append((color_prec_frame))

    images_per_row = 2

    #cv2.imshow( "ALL_1", np.hstack(imgs))
    cv2.imshow( "ALL_1", imgs)

    # show images
    height = sum(image.shape[0] for image in imgs)
    width = max(image.shape[1] for image in imgs)
    output = np.zeros((height, width, 3))


    ####################################################################################################
    # plot color percetage progration
    ####################################################################################################

    if print_color_prog == 1:
        numpy_color = np.array(color_pers_list[-500:])

        if frame_num == 0:
            print("creating figure for color change")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            fig.canvas.set_window_title('broken spiral')
            lines = []
            for color ,th in colors:
                color_scaled = [x/255 for x in color]
                rgb_color = tuple(color_scaled)
                lines.append(ax.plot([], [], color=rgb_color, label=str(color)))

            plt.grid(color='k', linestyle='-', linewidth=0.1)
            ax.legend()

        for i in range(0, len(colors)):
            A = numpy_color[:,i]
            x = lines[i][0]
            x.set_ydata(A)
            x.set_xdata(range(len(A)))

        ax.relim()
        ax.autoscale_view()
        plt.ylim(0, 100)
        plt.pause(0.01)
        if save_perc_graph == 1:
            plt.savefig('foo.png')

    ####################################################################################################
    # send osc message
    ####################################################################################################

    # send osc messages
    play_synth(color_prec_frame)

    ####################################################################################################
    # show origranem frame
    ####################################################################################################

    if show_org == 1:
        cv2.imshow('org', frame2)

    ####################################################################################################
    # plot histogram
    ####################################################################################################

    if histogram == 1:
        color = ('b', 'g', 'r')
        histr = None
        plt.ion()
        for channel, col in enumerate(color):
            histr = cv2.calcHist([frame2], [channel], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
            plt.ylim([0,15000])

        plt.grid(color='k', linestyle='-', linewidth=0.1)

        plt.title('Histogram for color scale picture')
        plt.pause(0.01)
        plt.gcf().clear()


    ####################################################################################################
    # calc frame rate and rewrite iteration and keyboard input
    ####################################################################################################

    #  calc frame rate
    if frame_num%10 == 0:
        elapsed = time.time() - t
        print("fps: " + str(10/elapsed))
        t = time.time()

    # keyboard input
    k = cv2.waitKey(30) & 0xff     # keyboard messages
    if k == 27:
        break   # break loop
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2) # write frame
        cv2.imwrite('opticalhsv.png',bgr)

    # next iteration
    prvs = next
    frame_num += 1

####################################################################################################
# end programm
####################################################################################################

cap.release()
cv2.destroyAllWindows()
