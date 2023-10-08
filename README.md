# W.A.K.E Documentation

This repository contains code and documentation for Wearable Alertness and Kinesis Equipment (W.A.K.E).

**Team Members**: Jarod Marshel, Tony Fu, Chonghao Cai, Chenhan Dai, Alessandro Bifulco

---

## Introduction

Our team developed this project for the NeuroTEC x Synaptech 2023 Hackathon. The hackathon lasted three days and focused on neurotechnology prototypes. We used the MUSE 2 headband and the TENS 3000 unit. Our goal was to help people stay awake by detecting changes in brainwaves and stimulating the body.

*Note: I wrote this documentation months after the event, so some details are missing.*

![Hackathon Poster](./images/hackathon_poster.png)

**Spoiler**: We didn't win the hackathon. The winning project, which also used a MUSE 2 headband and an Oculus Quest 3, stood out for its creative use of VR. Our project raised concerns about resembling a torture device, a critique we found reasonable.

[View Presentation Slides](https://prezi.com/view/d1VaiqtHbyYKQhuBqlw4/)

---

## Hardware Components

### MUSE 2 Headband

The MUSE 2 headband comes with 2 active EEG sensors (AF7 and AF8) and 3 reference sensors (Fp1, Fp2, FpZ) on the forehead and 2 sensors behind the ears (TP9 and TP10). We can also optionally plug in an auxiliary sensor (AUX_RIGHT) to the micro-USB port.

![MUSE 2 Headband](./images/muse2_headband.png)

Figure source: [MUSE 2 Specs](https://ifelldh.tec.mx/sites/g/files/vgjovo1101/files/Muse_2_Specifications.pdf)

To connect the MUSE 2 to a computer, you need a BLED112 USB dongle ([Here is the one I use](https://www.digikey.com/en/products/detail/silicon-labs/BLED112-V1/4245505)) because most computers lack stable BLE support.

### TENS 3000 Unit

The TENS 3000 unit is a pain relief device. It sends small electric currents through electrodes placed on the skin. We used it because its tingling sensation can help keep people awake.

![TENS 3000](./images/tens3000.jpg)

You can buy it from the [TENS Pro Official Website](https://www.tenspros.com/tens-3000-analogue-tens-unit-dt3002.html).

### Arduino Uno and Servo Motors

The TENS 3000 only has manual dials for electrical intensity. We used an Arduino Uno and servo motors to turn these dials. We built a cardboard housing to maintain pressure on the dials.

---

## Software Details

Please allow me to present this section in the form of a tutorial.

### Step 1. Connecting to the MUSE 2

For those who wants to develop their own MUSE 2 applications, please refer to [this repository by NueroTechX](https://github.com/NeuroTechX/bci-workshop) for starter code and tutorials. In short, we will use the following Python libraries:

- `pylsl`: the interface to the Lab Streaming Layer (LSL), a protocol for real-time streaming of biosignal over a network. Here is their [documentation](https://labstreaminglayer.readthedocs.io/). Here are some [example code](https://github.com/labstreaminglayer/pylsl/tree/master/pylsl/examples). Here is a [YouTube tutorial on LSL](https://youtu.be/Y1at7yrcFW0?si=V298gu2gYSO6tr3a).
- `sklearn`: for machine learning
- `scipy`: for signal processing
- `seaborn`: for data visualization
- Other libraries for connecting via the BLED112 USB dongle: `pygatt`, `bitstring`, `pexpect`. You can install those three libraries with the following command:

```bash
pip install git+https://github.com/peplin/pygatt pylsl bitstring pexpect
```

To connect to the MUSE 2, you need to first pair the MUSE 2 with your computer. You can do this with the following command:

```bash
python3 muse-lsl.py --name <YOUR_DEVICE_NAME>
```

The `<YOUR_DEVICE_NAME>` is the name of your MUSE 2. This information could be found on the box of your MUSE 2. For example, my MUSE 2's name is `Muse-E02A`. This command will establish a connection between your computer and the MUSE 2. Closing this window will result in a disconnection. If the program is complaining about not finding the MUSE 2: try restarting the MUSE 2, double-check the address, and make sure the MUSE 2 is not connected to any other devices.

#### Under the Hood of `muse-lsl.py`

The purpose of `muse-lsl.py` is to put our MUSE 2 device on the LSL network, so that other programs can fetch the device's data from LSL.

At the start of the program, we need to import the LSL library.

```python
from pylsl import StreamInfo, StreamOutlet, local_clock
```

Then, we need to construct a new [`StreamInfo`](https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/streaminfo.html?highlight=streaminfo) object.

```python
info = info = StreamInfo(name='Muse', type='EEG', channel_count=5, nominal_srate=256, channel_format='float32', source_id=options.name)  # name is command line argument, e.g. Muse-E02A
```

The `name` can actually be anything you want. It is just used to find the stream on the LSL network. The `type` is the type of data we are streaming. Refer to this page to see [all the available types](https://github.com/sccn/xdf/wiki/Meta-Data). We have 5 channels. All have a sampling rate of 256 Hz, and the format is `float32`.

The `StreamInfo` object provides a `desc()` method that allow the user to extend the metadata of the stream, so that other programs can understand the data. For example, we can add the manufacturer of the device and the labels of the channels.

```python
info.desc().append_child_value("manufacturer", "Muse")
channels = info.desc().append_child("channels")

for c in ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']:
    channels.append_child("channel") \
        .append_child_value("label", c) \
        .append_child_value("unit", "microvolts") \
        .append_child_value("type", "EEG")
```

So far, we only created a `StreamInfo` object, but how to make this stream available on the LSL network? We need to create a [`StreamOutlet`](https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/outlet.html) object. We need to pass the `StreamInfo` object to the constructor of `StreamOutlet`. Also, for performance reasons, let's make the stream "upload" data in chunks of 12 samples, and buffer up to 6 minutes (360 seconds) of data.

```python
outlet = StreamOutlet(info=info, chunk_size=12, max_buffered=360)
```

The rest of the code involves connecting the MUSE 2 to the `StreamOutlet`. This is better handled using an object-oriented approach. We will create a `Muse` class that handles the connection. It is has four public methods:

- `connect()`: find the MUSE 2, connect to it (using `pygatt`), subscribe to its channels (UUIDs), and register a callback function to handle the data.
- `start()`: initialize the data and timestamp variables.
- `stop()`: stop the data stream.
- `disconnect()`: disconnect from the MUSE 2.


```python
muse = Muse(callback=process, time_func=local_clock, name=options.name) # name is command line argument, e.g. Muse-E02A

muse.connect()
muse.start()

while 1:
    try:
        sleep(1)
    except:
        break

muse.stop()
muse.disconnect()
```

The `process` function is the callback function that handles the data. It uses the `StreamOutlet.push_sample()` method to push the data to the LSL network 12 samples at a time.

```python
def process(data, timestamps):
    for ii in range(12):
        outlet.push_sample(data[:, ii], timestamps[ii])
```


---


### Step 2. "Hello World": Data Streaming

In Step 1, we established a connection between the MUSE 2 and the computer. Specifically, we put the EEG data from our MUSE 2 on the LSL network. Now, we need to fetch the data from the LSL network. We will use the `pylsl` library again.

Open a new terminal window and run the following command:

```bash
python3 exercise_01.py
```

This will run the `exercise_01.py` program provided by NeuroTechX (also available in this repository). This program will plot the data from the MUSE 2. You should see something like this:

![exercise_01_output](./images/exercise_01_output.gif)

We are plotting the left ear (TP9) EEG data. The left plot shows the raw signal, and the right shows the signal being broken down into four frequency bands: delta (< 4 Hz), theta (4 - 8 Hz), alpha (8 - 12 Hz), and beta (12 - 30 Hz). The three spikes in the raw signal correspond to the three blinks I made while recording the data.

#### Under the Hood of `exercise_01.py`

##### Step 2.1 Connecting to the EEG stream

We will first import the LSL library.

```python
from pylsl import StreamInlet, resolve_byprop
```

The `resolve_byprop()` function will find the stream on the LSL network. We will use the `name` property to find the stream. Since there is only one stream on the network, we can just use the first element of the list returned by `resolve_byprop()`.

```python
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')

inlet = StreamInlet(streams[0], max_chunklen=12)
```

We need to apply a [time correction](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html?highlight=time%20correction#synchronization-process-offset-correction) to the data. This is because the data is not synchronized with the computer's clock.

```python
_ = inlet.time_correction()
```

Of course, we have access to the `StreamInfo` object that we defined in `muse-lsl.py`. We can use it to get the sampling rate of the data.

```python
info = inlet.info()
fs = int(info.nominal_srate())  # 256 Hz
```

##### Step 2.2 Set plot parameters


