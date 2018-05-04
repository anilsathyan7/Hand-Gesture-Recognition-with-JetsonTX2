# Hand-Gesture-Recognition-with-JetsonTX2
A small project for hand gesture recognition on NVIDIA Jetson TX2 using Deep learning and Caffe.It is basically a classification program to recognize images
using a webcam,in real-time.The dataset used for trainig contains 32,000 images labelled across the classes '0-5' of numbers (i.e in finger/palm).

## Getting Started

Recommended basic knowledge of Embedded Systems and Linux.

## Prerequisites

Nvidia Jetson TX2 flashed with Jetpack and host PC with Ubuntu 16.04 and a Webcam.
( JetPack 3.1 with L4T R28.1 used for this setup & experiment. )

Seven Segment Display (CA),LED's (R,G,B,O colours), Resistors (120 ohm), Jumper wires, Hook up wires,Breadboard etc.

Additionally, it requires Caffe and OpenCV3 to be installed in jetson TX2.

If you want to train on your own data, use Caffe and NVIDIA Digits on PC.

### Installing

See: https://developer.nvidia.com/embedded/jetpack for installation of Jetpack.

See: https://github.com/jetsonhacks/installCaffeJTX2 for installation of Caffe.

See: https://jkjung-avt.github.io/opencv3-on-tx2/ for installation of OpenCV

### Running the tests

First, connect the Seven Segment Display to GPIO pins as shown in video through resistors.
Ensure proper 'ground' (GND) connections.The required data files for the program can be found in data folder.Make neccessary changes for file path in the program before executing the program.
Enure the dependedncies and packages are properly installed (versions).

In terminal:-

Login as root and execute the python scripts after ensuring the appropriate connections.

Examples:-
$ python gesture_cv.py


## Versioning

Version 1.0

## Authors

Anil Sathyan
## License

Free to use, share or modify!! ... Copyleft!!

## Acknowledgments
* "http://www.jetsonhacks.com/nvidia-jetson-tx2-j21-header-pinout/"
* "https://youtu.be/D7lkth34rgM"
* "https://developer.nvidia.com/embedded/twodaystoademo"
* "http://www.jetsonhacks.com/2015/12/29/gpio-interfacing-nvidia-jetson-tx1/"
* "https://github.com/jgv7/CNN-HowManyFingers"
*  Nvidia Developer Forums - "https://devtalk.nvidia.com/"
