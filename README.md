# MC2_gesture_detection_pynqz2_tensorflow

Runing gesture detection on Pynq-z2 board with  tensorflow2.3.0
 source code which compiled by ourself.

```
sudo -i
sudo apt-get update
```

# install tensorflow2.3.0

## install python3.7
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













