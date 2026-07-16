# Raspberry Pi 4 and Coral USB Accelerator

<img width="4096" height="2304" alt="image" src="https://github.com/user-attachments/assets/e34e0443-ae7b-4c61-9620-370b1b52cc8a" />

**Disclaimer**: pre-trained model and labels belongs to their respective owners, original link: https://gweb-coral-full.uc.r.appspot.com/models/object-detection.

## Usage

### Library, Tools and Setup 

- Setup Raspberry Pi 4 Buster version: https://downloads.raspberrypi.com/raspios_arm64/images/raspios_arm64-2020-08-24/
- Setup APT package manager to point to the official legacy archive:
```sh
sudo nano /etc/apt/sources.list

### Comment out dead links
#deb http://deb.debian.org/debian buster main contrib non-free
#deb http://deb.debian.org/debian-security/ buster/updates main contrib non-free
#deb http://deb.debian.org/debian buster-updates main contrib non-free

### Add these 2 like below
deb http://archive.debian.org/debian buster main contrib non-free
deb http://archive.debian.org/debian-security buster/updates main contrib non-free

# Uncomment deb-src lines below then 'apt-get update' to enable 'apt-get source'
#deb-src http://deb.debian.org/debian buster main contrib non-free
#deb-src http://deb.debian.org/debian-security/ buster/updates main contrib non-free
#deb-src http://deb.debian.org/debian buster-updates main contrib non-free
```

```sh
sudo nano /etc/apt/sources.list.d/raspi.list

deb http://archive.raspberrypi.org/debian/ buster main
```
- Setup Coral USB Accelerator on Raspberry Pi 4: https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started
- Install necessary python3 libraries:
    - `sudo apt install python3-matplotlib`
    - `sudo apt install python3-numpy`
    - `sudo apt install cpufrequtils` - benchmark utility, it is used to set CPU frequency

### run_vd.py

Process a video using object detection, then generate a video output. Simulate real-time camera processing by writing back the same frame if the inference process consume more than allowed time period of a frame - creating 'lag' in video processing.

```sh
python3 run_video.py <model> <labels> <input_video> <output_video> [cpu_thread] [confidence] [no_gen_out]")

<model>:                  If string contain the 'edgetpu' substring then use edge tpu device else use cpu.
[cpu_thread] [confidence] is optional, default is 1 thread and 0.5, cpu thread is used when we run CPU model.
[no_gen_out]              is optional, input `NOGEN` to stop generating output video.
```

### benchmark.py

Run benchmark on various model under `model/` and generate a linechart as output. This will take sometime depends on your video input and how many model you wish to run. Currently, the code will run 3 times, each time increase the CPU thread by 1 to benchmark CPU performance against TPU.

```sh
python3 benchmark.py
```

**Note**: Cores running at 600 MHz.

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/b9c01c9d-56e8-4c59-b57e-ee36287b5fb5" />

**Note**: Cores running at 1.5 GHz.

<img width="1500" height="900" alt="image" src="https://github.com/user-attachments/assets/5c7abd93-e74f-4b28-9f3f-c7975e492c30" />


### Misc

Sample label file `coco_labels.txt` and video `in_vid.mp4`
