import os
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop

import bci_workshop_tools as BCIw

SERIAL_PORT = '/dev/cu.usbmodem1101'  # modify this to your serial port!

if __name__ == "__main__":

    """ 0. INITIALIZE SERIAL PORT """
    arduino = serial.Serial(port=SERIAL_PORT, timeout=0)
    time.sleep(2)

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL stream
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info, description, sampling frequency, number of channels
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    n_channels = info.channel_count()

    # Get names of all channels
    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, n_channels):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    """ 2. SET EXPERIMENTAL PARAMETERS """

    buffer_length = 15      # in seconds
    epoch_length = 1        # in seconds
    # Overlap between two consecutive epochs (in seconds)
    overlap_length = 0.8

    # Amount to 'shift' the start of each next consecutive epoch
    shift_length = epoch_length - overlap_length

    # Index of the channel (electrode) to be used
    index_channel = [0, 1, 2, 3]
    ch_names = [ch_names[i] for i in index_channel]
    n_channels = len(index_channel)

    feature_names = BCIw.get_feature_names(ch_names)

    """ 3. RECORD TRAINING DATA """
    training_length = 20  # in seconds

    # Record data while eyes are open
    os.system(f"say 'Open your eyes'")
    eeg_data0, timestamps0 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data0 = np.array(eeg_data0)[:, index_channel]

    # Record data while eyes are closed
    os.system(f"say 'Close your eyes'")
    eeg_data1, timestamps1 = inlet.pull_chunk(
            timeout=training_length+1, max_samples=fs * training_length)
    eeg_data1 = np.array(eeg_data1)[:, index_channel]

    # Divide data into epochs
    eeg_epochs0 = BCIw.epoch(eeg_data0, epoch_length * fs,
                             overlap_length * fs)
    eeg_epochs1 = BCIw.epoch(eeg_data1, epoch_length * fs,
                             overlap_length * fs)

    """ 4. COMPUTE FEATURES AND TRAIN CLASSIFIER """

    feat_matrix0 = BCIw.compute_feature_matrix(eeg_epochs0, fs)
    feat_matrix1 = BCIw.compute_feature_matrix(eeg_epochs1, fs)

    [classifier, mu_ft, std_ft] = BCIw.train_classifier(
            feat_matrix0, feat_matrix1, 'SVM')

    os.system(f"say 'Training complete!'")

    """ 5. USE THE CLASSIFIER IN REAL-TIME"""

    # Initialize the buffers for storing raw EEG and decisions
    eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
    filter_state = None  # for use with the notch filter
    decision_buffer = np.zeros((30, 1))

    plotter_decision = BCIw.DataPlotter(30, ['Decision'])

    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(shift_length * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = BCIw.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

            """ 3.2 COMPUTE FEATURES AND CLASSIFY """
            # Get newest samples from the buffer
            data_epoch = BCIw.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Compute features
            feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
            y_hat = BCIw.test_classifier(classifier,
                                         feat_vector.reshape(1, -1), mu_ft,
                                         std_ft)
            print(y_hat)
            if int(y_hat[0]) == 1: # eyes close, apply TENS
                arduino.write(str.encode(f"3, 3"))
            else:
                arduino.write(str.encode(f"0, 0"))

            decision_buffer, _ = BCIw.update_buffer(decision_buffer,
                                                    np.reshape(y_hat, (-1, 1)))

            """ 3.3 VISUALIZE THE DECISIONS """
            plotter_decision.update_plot(decision_buffer)
            plt.pause(0.00001)

    except KeyboardInterrupt:

        print('Closed!')
