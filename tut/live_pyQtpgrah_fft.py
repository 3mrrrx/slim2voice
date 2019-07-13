#python script to read 64 bytes of data from tiva C and plot them
#using pyQtGraph on a loop. As soon as 64 bytes arrive plot is updated


from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np


import serial

ser = serial.Serial(
    port='COM4',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

ser.close()
ser.open()
print (ser.isOpen())

data = np.zeros(64)

app = QtGui.QApplication([])

p = pg.plot()
p.setWindowTitle('FFT')
p.setLabel('bottom', 'Index', units='B')
curve = p.plot()


def update():
    global curve, data,
    if(ser.readline() == b'*\n'):   # "*" indicates start of transmission from tiva c
        for h in range(63):
            try:
                data[h] = int(ser.readline())  #get 64 bytes from tiva c and plot them
            except:
                pass
        curve.setData(data)
        app.processEvents()  ## force complete redraw for every plot



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
