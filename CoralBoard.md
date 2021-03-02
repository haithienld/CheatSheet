### 1 Insert SD card

### 2 Now get ID of SD card

sudo fdisk -l
<various disks, snipped>
Disk /dev/mmcblk1: 116.2 GiB, 124721823744 bytes, 243597312 sectors
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
So in my case, the SD card is /dev/mmcblk1

### 3 Format disk
sudo mkfs.ext4 /dev/mmcblk1
Now wait a while while it formats.

Mount newly formatted disk and prepare to migrate existing home
# mount it to /mnt 
sudo mount /dev/mmcblk1 /mnt
