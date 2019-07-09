# slim2voice

this is a python script to convert the into from a web came or video into OSC messages. The script take in a array of RGB color vectors ([R,G,B],[R,G,B]), and find the percentage of that color in the image. Each percentage can be scaled and translated in to an OSC message.

The repository includes an Supercollider Sketch for testing the OSC messages.

## Dependencies

The script uses python 3.6, and requires the following libraries:
* OpenCV
* matplotlib
* numpy
* pythonosc

on a linux system these files can be installed using the following code:
```bash
pip3 install opencv-python matplotlib numpy python-osc
```


### Example

To run the code one has to update the scanned color vector and run the code.

```python
colors = [([120, 0, 0], 100), ([0, 120, 0], 100), ([135,4,30], 100), ([0, 120, 0], 100), ([150, 0, 180], 100),([130, 0, 130], 100),([155,124,0], 100)]
```
One color in the color Vector is defined by the tuple of the RGB values and an Delta window:
```python
color  = ([120, 0, 0], 100) # ([R+Delta,G+Delta,B+Delta], Delta)
```

run the script using the following command :
```bash
python3 color2osc
```

by default the code will use the web cam, other wise the path should be set for video.

Different debugging flags can be set at the beginning of the script:

```python
histogram = 0           # plots a live histogram of the colors
show_org = 0            # shows a live image from orginal video
motion_dictection = 0   # not implanted
save_image = 0          # saves the recorded video to creat loops (not full implanted)
save_perc_graph = 0     # saves the finale version of the recorded histogram of the colors
print_color_prog = 1    #
```


### Supercollider Example

To install and run Supercollider please refer to the supercollider [webpage](https://supercollider.github.io/download).

the sketch includes 14 triangle wave oscillators that are tuned using micro tonal scale based on a fundamental frequency (default: f_fund = 440 //Hz). The color2osc_14notes.py code will send osc messages to control 14 resonant low pass filters of each of oscillators.

To run the example start the python script:    

```bash
python3 color2osc_14notes.py
```
the codes uses by default the following colors:

```python
colors = [([120, 0, 0], 100), ([0, 120, 0], 100), ([135,4,30], 100), ([0, 120, 0], 100), ([150, 0, 180], 100),([130, 0, 130], 100),([155,124,0], 100)]
```

(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)

then start the supercollider sketch:

```
until finished
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgement

The name for this project comes form looking at slimy color changes under the lens of   
