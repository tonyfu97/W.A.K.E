"""
Exercise 1: A neurofeedback interface (single-channel)
======================================================

Description:
In this exercise, we'll try and play around with a simple interface that
receives EEG from one electrode, computes standard frequency band powers
and displays both the raw signals and the features.

"""

import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop

import bci_workshop_tools as BCIw


if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL stream
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    _ = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    desc = info.desc()
    fs = int(info.nominal_srate())

    # Get the channels names
    labels = []
    channels = desc.child("channels").first_child()
    while not channels.empty():
        labels.append(channels.child_value("label"))
        channels = channels.next_sibling()

    """ 2. SET EXPERIMENTAL PARAMETERS """

    buffer_length = 15      # in seconds
    epoch_length = 1        # in seconds
    # Overlap between two consecutive epochs (in seconds)
    overlap_length = 0.8

    # Amount to 'shift' the start of each next consecutive epoch
    shift_length = epoch_length - overlap_length

    # Index of the channel (electrode) to be used
    index_channel = [0]
    ch_names = [labels[i] for i in index_channel]

    # Get names of band frequencies (e.g., 'alpha', 'beta')
    feature_names = BCIw.get_feature_names(ch_names)

    """ 3. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer (for plotting)
    eeg_buffer = np.zeros((int(fs * buffer_length), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length" (used for plotting)
    n_win_test = int(np.floor((buffer_length - epoch_length) /
                              shift_length + 1))

    # Initialize the feature data buffer (for plotting)
    feat_buffer = np.zeros((n_win_test, len(feature_names)))

    # Initialize the plots
    plotter_eeg = BCIw.DataPlotter(fs * buffer_length, ch_names, fs)
    plotter_feat = BCIw.DataPlotter(n_win_test, feature_names,
                                    1 / shift_length)

    """ 4. GET DATA """

    print('Press Ctrl-C in the console to break the while loop.')

    try:
        while True:

            """ 4.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(shift_length * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, index_channel]

            # Update EEG buffer
            eeg_buffer, filter_state = BCIw.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 4.2 COMPUTE FEATURES """
            # Get newest samples from the buffer
            data_epoch = BCIw.get_last_data(eeg_buffer,
                                            epoch_length * fs)

            # Compute features
            feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
            feat_buffer, _ = BCIw.update_buffer(feat_buffer,
                                                np.asarray([feat_vector]))

            """ 4.3 VISUALIZE THE RAW EEG AND THE FEATURES """
            plotter_eeg.update_plot(eeg_buffer)
            plotter_feat.update_plot(feat_buffer)
            plt.pause(0.00001)

    except KeyboardInterrupt:
        print('Closing!')
