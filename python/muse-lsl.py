from muse import Muse
from time import sleep
from pylsl import StreamInfo, StreamOutlet, local_clock
from optparse import OptionParser

CHUNK_SIZE = 12

parser = OptionParser()
parser.add_option("-n", "--name",
                  dest="name", type='string', default="Muse-E02A",
                  help="name of the device.")

(options, args) = parser.parse_args()

info = StreamInfo(name='Muse',
                  type='EEG',
                  channel_count=5,
                  nominal_srate=256,
                  channel_format='float32',
                  source_id=options.name)

info.desc().append_child_value("manufacturer", "Muse")
channels = info.desc().append_child("channels")

for c in ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']:
    channels.append_child("channel") \
        .append_child_value("label", c) \
        .append_child_value("unit", "microvolts") \
        .append_child_value("type", "EEG")

outlet = StreamOutlet(info=info, chunk_size=CHUNK_SIZE, max_buffered=360)


def process(data, timestamps):
    for ii in range(CHUNK_SIZE):
        outlet.push_sample(data[:, ii], timestamps[ii])


muse = Muse(callback=process, time_func=local_clock, name=options.name)

muse.connect()
muse.start()

while 1:
    try:
        sleep(1)
    except:
        break

muse.stop()
muse.disconnect()
