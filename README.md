# PanNote

This repository have been build with the contribution of [sepideh-shamsizadeh](https://github.com/sepideh-shamsizadeh) and [leobarcellona](https://github.com/leobarcellona). 
For any issues or problems, feel free to contact us. 

**Our paper has been accepted at 2024 IEEE International Conference on Robotics and Automation (ICRA 2024)**

# Easy Start Up

Here you will find some quick startup examples. 

## 1. Collect data 

To collect data, you can use our launch file (tested on ROS Noetic). You may need to update paths and topic names.
```
roslaunch auto_calibration_tools acquireCalibrationData.launch
```
We advise to use a big size ball (radius > 50 cm). We integrate an audio feedback to facilitate the acquisition. For better accuracy, we advise to listen to the audio feedback and stop moving just before the acquisition of the data. 

## 2. Train the ball detector
The ball is detected automatically in the image wih one shot detector. You can train it with:

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 train_ball_detector.py
```
The scripts read images from `backgrounds_UHD/` folder and `target_ball.png` to train a one-shot detector. Default test folder is `images_UHD_ball_indoor2/`.

## 3. Run calibration with the ball

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 processCalibrationBall.py
```
This step extracts features from each image and laser scan couple. The results are saved into `cameraLaser_pointsUHD_ball_pano_i.pkl`. The data are read from `imagesUHD_ball_400i/` folder.

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 calibrateCamera2LaserPnP.py
```
is used to calibrate the panoramic image with the laser. The final output is a dictionary like: 

```python
    results = {
        "H2":H2,
        "H3":H3
    }
```
saved into `laser2camera_map.pkl`. You can use H2 to feed the function `projectPoint2Image(laser_point, H)` or H3 with the function `predictU3(H, x_3d, y_3d)` to map laser points into the panoramic image. 

## 4. Collect data for labelling
To collect data to be labelled, you can use our launch file (tested in ROS Noetic). Ensure the camere is turned on. You may need to update paths and topic names.
```
roslaunch auto_calibration_tools recordBag.launch bag_filename:=bag_name
```
Then, you process your own bag with:
```
roslaunch auto_calibration_tools extract_from_bag.launch save_step:=15 save_dir:=folder
```
The processing will extract images and laser scans in `auto_calibration_tools/bag_extraction/folder`.

## 5. Automatic label the data
To annotate the images extract in step 4, you can run: 
```commandline
cd AutoLabeling/src/auto_labelling_tools
python3 main.py
```
You may need to update the paths. Annotations are saved in `out/automatic_annotations.csv` file. 

## Train a baseline model and test 
```commandline
cd AutoLabeling/src/auto_labelling_tools
python3 train_2.py
```
To train the model, you can feed the folders containing the labelled data. You may need to update paths. Test runs automatically after training in the specified folder. 
___________________________________________

In the following more detailed instrutions on how to use the repository. 


# STARTUP THE CAMERA

## Install drivers (only for 1st installation)

This steps are takes from this [guide](https://husarion.com/tutorials/ros-components/ricoh-theta-z1/). 

### v4l2loopback

First download, build and install v4l2loopback. It will allow you to create virtual loopback camera interfaces.

```
mkdir -p ~/your_ws/src/theta_z1
cd ~/your_ws/src/theta_z1/
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback/
make && sudo make install
sudo depmod -a
```

After successful installation run:

```
ls /dev | grep video
```

You should see your video interfaces.

If you don't have any other cameras installed the output should be empty. To start loopback interface and find it's ID run:

```
sudo modprobe v4l2loopback
ls /dev | grep video
```
New /dev/video device should appear. It's your loopback interface you will later assign to your THETA Z1.

**ATTENTION**: if you get an error like
```
rmmod: ERROR: Module v4l2loopback is not currently loaded
modprobe: ERROR: could not insert 'v4l2loopback': Operation not permitted
```

Try to solve it with
```
sudo apt-get install v4l2loopback-dkms
```
### Ricoh Theta dependencies

Install required packages:

```

sudo apt-get install libgstreamer1.0-0 \
     gstreamer1.0-plugins-base \
     gstreamer1.0-plugins-good \
     gstreamer1.0-plugins-bad \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-libav \
     gstreamer1.0-doc \
     gstreamer1.0-tools \
     gstreamer1.0-x \
     gstreamer1.0-alsa \
     gstreamer1.0-gl \
     gstreamer1.0-gtk3 \
     gstreamer1.0-qt5 \
     gstreamer1.0-pulseaudio \
     libgstreamer-plugins-base1.0-dev \
     libjpeg-dev
```

After installation building and install libuvc-theta:

```
cd ~/your_ws/src/theta_z1
git clone https://github.com/ricohapi/libuvc-theta.git
cd libuvc-theta
mkdir build
cd build
cmake ..
make
sudo make install
```

### Installation
```
git clone https://github.com/bach05/libuvc-theta-sample.git
cd gst
make
```

## Run the camera

```
cd your_workspace/src/ricoh_theta_z1/src/libuvc-theta-sample/gst
sh streaming_theta.sh
```
```
export ROS_MASTER_URI=http://192.168.53.2:11311
export ROS_IP=192.168.53.10
rosrun auto_calibration_tools camera.py
```

# CALIBRATION

## Intrinsics

### Collect data
Connect the PC to the robot and the camera. Start up the camera. You can use:
```
roslaunch auto_calibration_tools acquireCalibrationData.launch
```
Be careful to set `SAVE_ROOT`  to `~/AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics/images` in the launch file. 
Move around with the chessboard. Default is a [] chessboard.

### Train the board detector

The chessboard is detected automatically in the image to remap it in the right side. You can easily train a detector for your chessboard

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics
python3 train_board_detector.py
```
The scripts read images from `backgrounds/` folder and `target.png` to train a one-shot detector. 

### Run calibration

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/calibration_data_intrinsics
python3 splitImage.py
```
This step split each panoramic image in the 6 cube projections. It reads the panoramic images from `/images`.

```commandline
python3 calibrateIntrinsics.py
```
gives the intrinsics calibration parameters for each view of the cube projection in `intrinsics.pkl`

## Camera-Laser Extrinsics

### Collect data
Connect the PC to the robot and the camera. Start up the camera. You can use:
```
roslaunch auto_calibration_tools acquireCalibrationData.launch
```
Be careful to set `SAVE_ROOT`  to `~/AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration/images_outdoor` in the launch file. 

Move the robot around the board. We suggest to do this in an empty room or in outdoor to avoid spourious detections. 

### Train the board detector

The chessboard is detected automatically in the image to remap it in the correct side. You can easily train a detector for your chessboard

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 train_board_detector.py
```
The scripts read images from `backgrounds/` folder and `target.png`,`target2.png` to train a one-shot detector. 

### Train the board detector

The chessboard is detected automatically in the image to remap it in the correct side. You can easily train a detector for your chessboard

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 train_ball_detector.py
```
The scripts read images from `backgrounds_UHD/` folder and `target_ball.png` to train a one-shot detector. Test folder is `images_UHD_ball_indoor2/`

### Run calibration with the board

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 processCalibrationDataPano.py
```
This step extracts features from each image and laser scan couple. The results are saved into `cameraLaser_pointsUHD_pano_indoor2.pkl`.  The data are read from `images_UHD_indoor/` folder.

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 calibrateLaser2CameraPolar.py
```
is used to calibrate the panoramic image with the laser. The final output is a dictionary like: 

```python
    results = {
        "ransac":R_params
    }

```
saved into `laser2camera_polar_map.pkl`. You can use it to feed the function `predRANSAC(ransac_params, rho_laser, theta_laser)` to map laser points into the panoramic image. 


### Run calibration with the ball

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 processCalibrationBall.py
```
This step extracts features from each image and laser scan couple. The results are saved into `cameraLaser_pointsUHD_ball_pano_i.pkl`. The data are read from `imagesUHD_ball_400i/` folder.

```commandline
cd AutoLabeling/src/auto_calibration_tools/scripts/camera_laser_calibration
python3 calibrateCamera2LaserPnP.py
```
is used to calibrate the panoramic image with the laser. The final output is a dictionary like: 

```python
    results = {
        "K":K,
        "D":D,
        "T":tvec,
        "R":rvec,
        "H":H1,
        "z_value":z_value
    }
```
saved into `laser2camera_map.pkl`. You can use it to feed the function `projectPoint2Image(laser_point, K, D, T, R, H, z_value)` to map laser points into the panoramic image. 

# AutoLabeling
