#### Mout sdcard to Coral Board from https://github.com/f0cal/google-coral/issues/61
a. Insert SD card
b. Now get ID of SD card
sudo fdisk -l

c. So in my case, the SD card is /dev/mmcblk1   \
Format disk    \
sudo mkfs.ext4 /dev/mmcblk1


#### Swapfile from https://github.com/goruck/edge-tpu-servers/blob/master/README.md 
#### Make Swapfile
cd /media/

sudo mkdir mendel

sudo chmod a+rwx mendel or chmod 777 

cd mendel

edit /etc/fstab \
sudo nano /etc/fstab \ 
Paste \
/dev/mmcblk1 /media/mendel ext4 defaults 0 2   \
Reboot \
cd /media/mendel      \
Dont use this line - sudo mkdir swapfile    \
sudo dd if=/dev/mmcblk1 of=/media/mendel/swapfile bs=2M count=2048 oflag=append conv=notrunc \
sudo mkswap /media/mendel/swapfile  \
sudo swapon /media/mendel/swapfile \

test: free -h

(Optional) Edit your /etc/sysctl.conf and add:
vm.swappiness=10

#### Install zerorpc
sudo apt-get update

sudo nano ~/.bashrc \
export PATH=$PATH:/home/mendel/.local/bin:/sbin \
sudo apt install python3-dev libffi-dev

pip3 install zerorpc  #can insert --user
#### Important
sudo apt-get install python3-scipy python3-numpy python3-wheel

##### pip3 install --upgrade pip setuptools #important 
##### python3 -m pip install --upgrade --force-reinstall pip --user 

#### Plz view the link https://github.com/goruck/edge-tpu-servers/blob/master/README.md and OpenCV 4.4.5 https://qengineering.eu/install-opencv-4.5-on-raspberry-pi-4.html

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.0.zip  \
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.0.zip  \

unzip opencv.zip \
unzip opencv_contrib.zip \

#### Install lib
sudo apt-get update

sudo apt-get upgrade

sudo apt-get install cmake gfortran libjpeg-dev libtiff-dev libgif-dev libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libcanberra-gtk* libxvidcore-dev libx264-dev libgtk-3-dev libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev libhdf5-dev protobuf-compiler -y

#### If cannot use mdt shell please check connection between board and host. we can not using mdt shell when we use both wifi and OTG line

If Coral dont connect to Mornitor use
export DISPLAY=:0.0
https://stackoverflow.com/questions/25992088/linux-how-to-run-a-program-of-gtk-without-display-environment-gtk-warning
##### Need to see more zmq
https://github.com/zeromq/pyzmq
##### CheatSheet
Xoa mau xanh nen trong thu muc \
sudo chmod 0775 -R mendel/

### Tinker Edge T 
Reflash: sudo reboot-bootloader 2 times \
Connect by ssh: sudo /etc/ssh/sshd_config --> PasswordAuthentication no --> yes
https://chtseng.wordpress.com/2020/09/10/asus%E7%9A%84ai%E9%96%8B%E7%99%BC%E6%9D%BF%EF%BC%9Atinker-edge-t-board%E4%BB%8B%E7%B4%B9/
