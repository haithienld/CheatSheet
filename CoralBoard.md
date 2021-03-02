#### Mout sdcard to Coral Board from https://github.com/f0cal/google-coral/issues/61
a. Insert SD card
b. Now get ID of SD card
sudo fdisk -l

c. So in my case, the SD card is /dev/mmcblk1
Format disk
sudo mkfs.ext4 /dev/mmcblk1

#### Swapfile from https://github.com/goruck/edge-tpu-servers/blob/master/README.md 
#### Make Swapfile
cd /media/
sudo mkdir mendel
cd ../ 
sudo chmod 777 mendel
cd mendel
sudo mkdir swapfile
sudo dd if=/dev/mmcblk1 of=/media/mendel/swapfile bs=1M count=2048 oflag=append conv=notrunc
sudo mkswap /media/mendel/swapfile
sudo swapon /media/mendel/swapfile
test: free -h

(Optional) Edit your /etc/sysctl.conf and add:
vm.swappiness=10

#### Install zerorpc
sudo apt-get update
sudo apt install python3-dev libffi-dev
pip3 install zerorpc
#### Important
sudo apt-get install python3-scipy python3-numpy python3-wheel
pip3 install --upgrade pip setuptools #important
#### Plz view the link https://github.com/goruck/edge-tpu-servers/blob/master/README.md and OpenCV 4.4.5 https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html
#### Install lib
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cmake gfortran
sudo apt-get install libjpeg-dev libtiff-dev libgif-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libgtk2.0-dev libcanberra-gtk*
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev
sudo apt-get install libjasper-dev liblapack-dev libhdf5-dev
sudo apt-get install protobuf-compiler

#### If cannot use mdt shell please check connection between board and host. we can not using mdt shell when we use both wifi and OTG line

