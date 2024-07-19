import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, periodogram, filtfilt
from inference_preprocess import preprocess_raw_video, detrend


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def predict_vitals(args):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = './mtts_can.hdf5'
    batch_size = args.batch_size
    fs = args.sampling_rate
    sample_data_path = args.video_path

    dXsub = preprocess_raw_video(sample_data_path, dim=36, sample_dir=args.sample_dir)
    print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = filtfilt(b_resp, a_resp, np.double(resp_pred))

    ########## Plot ##################
    if args.fig_dir is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))
        ax1.plot(pulse_pred)
        ax1.set_title('Pulse Prediction')
        ax2.plot(resp_pred)
        ax2.set_title('Resp Prediction')
        plt.savefig(args.fig_dir)

    if args.pulse_dir is not None:
        with open(args.pulse_dir, 'wb') as pulse_file:
            np.save(pulse_file, pulse_pred)
    if args.resp_dir is not None:
        with open(args.resp_dir, 'wb') as resp_file:
            np.save(resp_file, resp_pred)

    bpm_fft = calculate_fft_hr(pulse_pred)
    print('BPM:', bpm_fft)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default=30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size (multiplier of 10)')
    parser.add_argument(
        '--sample-dir',
        type=str,
        default=None,
        help='where to save the sample pre-processed frame'
    )
    parser.add_argument(
        '--fig-dir',
        type=str,
        default=None,
        help='where to save the pulse and respiration figure'
    )
    parser.add_argument(
        '--pulse-dir',
        type=str,
        default=None,
        help='where to save the pulse vector'
    )
    parser.add_argument(
        '--resp-dir',
        type=str,
        default=None,
        help='where to save the respiration vector'
    )
    args = parser.parse_args()

    predict_vitals(args)
