import platform
from time import time
import numpy as np
import pygatt
import bitstring

CHUNK_SIZE = 12

class Muse:
    """Muse 2016 headband"""

    def __init__(self, callback=None, interface=None, time_func=time, name=None):
        self.address = None
        self.name = name
        self.callback = callback
        self.interface = interface
        self.time_func = time_func

        if platform == "linux" or platform == "linux2":
            self.backend = 'gatt'
        else:
            self.backend = 'bgapi'

    def connect(self):
        if self.backend == 'gatt':
            self.adapter = pygatt.GATTToolBackend(self.interface)
        else:
            self.adapter = pygatt.BGAPIBackend(serial_port=self.interface)
        self.adapter.start()

        self.address = self.find_muse_address(self.name)
        if not self.address:
            raise ValueError(f"Can't find Muse Device {self.name}")

        self.device = self.adapter.connect(self.address)
        self._subscribe_eeg()
        print('Connected')

    def find_muse_address(self, name=None):
        devices = {device['name']: device['address']
                   for device in self.adapter.scan(timeout=10.5)}
        if name:
            return devices.get(name)

        return next((addr for name, addr in devices.items() if 'Muse' in name), None)

    def start(self):
        """Start streaming."""
        self._init_sample()
        self.last_tm = 0
        self.device.char_write_handle(0x000e, [0x02, 0x64, 0x0a], False)
        print('Start streaming')

    def stop(self):
        """Stop streaming."""
        self.device.char_write_handle(0x000e, [0x02, 0x68, 0x0a], False)
        print('Stop streaming')

    def disconnect(self):
        """disconnect."""
        self.device.disconnect()
        self.adapter.stop()
        print('Disconnected')

    def _subscribe_eeg(self):
        """subscribe to eeg stream."""
        self.device.subscribe('273e0003-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0004-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0005-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0006-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)
        self.device.subscribe('273e0007-4c4d-454d-96be-f03bac821358',
                              callback=self._handle_eeg)

    def _unpack_eeg_channel(self, packet):
        """Decode data packet of one eeg channel.

        Each packet is encoded with a 16bit timestamp followed by 12 time
        samples with a 12 bit resolution.
        """
        aa = bitstring.Bits(bytes=packet)
        pattern = "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12, \
                   uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
        res = aa.unpack(pattern)
        timestamp = res[0]
        data = res[1:]
        # 12 bits on a 2 mVpp range
        data = 0.48828125 * (np.array(data) - 2048)
        return timestamp, data

    def _init_sample(self):
        """initialize array to store the samples"""
        self.timestamps = np.zeros(5)
        self.data = np.zeros((5, CHUNK_SIZE))

    def _handle_eeg(self, handle, data):
        """Calback for receiving a sample.

        sample are received in this order : 44, 41, 38, 32, 35
        wait until we get 35 and call the data callback
        """
        timestamp = self.time_func()
        index = int((handle - 32) / 3)  # [44, 41, 38, 32, 35] -> [4, 3, 2, 0, 1]
        tm, d = self._unpack_eeg_channel(data)

        if self.last_tm == 0:
            self.last_tm = tm - 1

        self.data[index] = d
        self.timestamps[index] = timestamp
        # last data received
        if handle == 35:
            if tm != self.last_tm + 1:
                print("missing sample %d : %d" % (tm, self.last_tm))
            self.last_tm = tm
            # affect as timestamps the first timestamps - 12 sample
            timestamps = np.arange(-CHUNK_SIZE, 0) / 256.
            timestamps += np.min(self.timestamps[self.timestamps != 0])
            self.callback(self.data, timestamps)
            self._init_sample()
