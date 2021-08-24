# MC2_gesture_detection_pynqz2_tensorflow

Runing gesture detection on Pynq-z2 board with  tensorflow2.3.0
 source code which compiled by ourself.

Precompile whl list:

Crosscompile on Docker
* tensorflow 2.3.0

Compile on Pynq-Z2
* numpy
* scipy
* h5py
* termcolor
* wrapt


# install tensorflow2.3.0

## install python3.7
```
sudo -i
sudo apt-get update
```

```
sudo apt-get install python3.7 python3.7-dev
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

sudo update-alternatives --config python3

sudo apt-get install  python-dev libhdf5-dev  python-apt 
python3.7 -m pip install pip
python3.7 -m pip install --upgrade pip

# check pip version
pip3.7 -V
```

## install depedency library
```
pip3.7 install Cython Jinja2 

ln -s /usr/lib/python3/dist-packages/apt_pkg.cpython-36m-arm-linux-gnueabihf.so /usr/lib/python3/dist-packages/apt_pkg.so

cd build_whl
pip3.7 install ./numpy-1.18.5-cp37-cp37m-linux_armv7l.whl
pip3.7 install ./h5py-2.10.0-cp37-cp37m-linux_armv7l.whl  # most after numpy
pip3.7 install ./scipy-1.4.1-cp37-cp37m-linux_armv7l.whl  # most after numpy

pip3.7 install ./termcolor-1.1.0-py3-none-any.whl
pip3.7 install ./wrapt-1.12.1-cp37-cp37m-linux_armv7l.whl

cd tensorflow_pkg_pynq
pip3.7 install ./tensorflow-2.3.0-cp37-none-linux_armv7l.whl
# patience wait :)
```
## Test tensorflow
```
# python3
import tensorflow as tf
print(tf.reduce_sum(tf.random.normal([1000, 1000])))
```

# Gesture detection
## install  dependency library
```
pip3.7 install pyserial
```
## run code

```
cd code
python3 real_time_detection.py
```
PS. loading model need about 5 minutes. be patience.



---


# Crosscompile Tensorflow
> 參考自
> https://github.com/lhelontra/tensorflow-on-arm
> https://github.com/franout/tensorflow_for_pynqz2

## Dependencies
On windows OS ， check below whether is installed or not.
```
WSL2 
Docker
```

## Crosscompile with Docker
Open WSL2(we use  Ubuntu 20.04.1 LTS)
```
#check docker 
docker -v


git clone https://github.com/lhelontra/tensorflow-on-arm

cd build_tensorflow/
docker build -t tf-arm -f Dockerfile .
docker run -it -v /tmp/tensorflow_pkg/:/tmp/tensorflow_pkg/ --env TF_PYTHON_VERSION=3.7 tf-arm

# if Crosscompile cause WSL2 unexcept error and shutdown
# tre below to limit cpu usage
# docker run -it -v /tmp/tensorflow_pkg_pynq/:/tmp/tensorflow_pkg/  --cpus="3"  --env TF_PYTHON_VERSION=3.7 tf-arm

```
in tf-arm container
```
# motify config
cd config/
nano rpi.conf
# look for 
# --copt=-mfpu=neon-vfpv4
# motify to
# --copt=-mfpu=neon
# save rpi.conf

# start Crosscompile
cd ../
./build_tensorflow.sh configs/rpi.conf

```
It may take long time and a lot of ram( We use 10 hour and 15G ram) 

Compile Finish will save in WSL2 `/tmp/tensorflow_pkg_pynq/`

### For other board
> 參考自 https://github.com/franout/tensorflow_for_pynqz2

if you have a different  board you can just check the cpu's flags with:
```
cat /proc/cpuinfo
```
and modify the copt accordingly.


# PreCompile whl

On PynqZ2 , if only do 
```
pip3.7 install ./tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```
may cause long time and a lot error.

The error cause by numpy will install after scipy and h5py.

Thus, we install twice

### first time
```
pip3.7 install ./tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```
take 5 hours 
Althouh installed but tensorflow is not working sucessful

### second time
clean cache and use --force-reinstall
```
pip3.7 cache purge
pip3.7 install --force-reinstall ./tensorflow-2.3.0-cp37-none-linux_armv7l.whl
```
take other 5 hours

Success install tensorflow

### looking for install message

* numpy
* scipy
* h5py
* termcolor
* wrapt

build successes  `.whl`  which will save in `.cache`

the path will show in install  message

cd to the path and copy out 
```
numpy-1.18.5-cp37-cp37m-linux_armv7l.whl
h5py-2.10.0-cp37-cp37m-linux_armv7l.whl  
scipy-1.4.1-cp37-cp37m-linux_armv7l.whl 
termcolor-1.1.0-py3-none-any.whl
wrapt-1.12.1-cp37-cp37m-linux_armv7l.whl
```












