#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/29 15:44
# @Author  : MengnanChen
# @File    : combine_sound.py
# @Software: PyCharm

import os
import glob
import random

from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 48000


def combine_rnnoise_contributions(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sound_files = glob.glob(os.path.join(input_dir, '*.raw'))
    random.shuffle(sound_files)
    print('There {} files, we just use 10 files'.format(len(sound_files)))
    sound_files = sound_files[:10]  # Just pick 1000 files
    output_sound = None
    for sound_file in tqdm(sound_files):
        y, sr = sf.read(sound_file, channels=1, samplerate=SAMPLE_RATE, endian='little', subtype='PCM_16')
        output_sound = np.concatenate((output_sound, y)) if output_sound is not None else y

    sf.write(os.path.join(output_dir, 'noise.raw'), output_sound, samplerate=SAMPLE_RATE, subtype='PCM_16')


def combine_uebin(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sound_files = glob.glob(os.path.join(input_dir, 'downsampled_22kHz', 'Mandarin_talkers', '*', '*', '*', '*.wav'))
    random.shuffle(sound_files)
    print('There {} files, we just use 100 files'.format(len(sound_files)))
    sound_files = sound_files[:100]  # Just pick 100 files
    output_sound = None
    for sound_file in tqdm(sound_files):
        y, sr = librosa.load(sound_file, sr=SAMPLE_RATE, mono=True, res_type='kaiser_fast')
        output_sound = np.concatenate((output_sound, y)) if output_sound is not None else y

    sf.write(os.path.join(output_dir, 'speech.raw'), output_sound, samplerate=SAMPLE_RATE, subtype='PCM_16',
             endian='LITTLE')


if __name__ == '__main__':
    # noise_input_dir = '../data/rnnoise_contributions'
    # combine_rnnoise_contributions(noise_input_dir, output_dir='../data/')

    speech_input_dir = '../data/UEDIN_mandarin_bi_data_2010'
    combine_uebin(speech_input_dir, output_dir='../data/')
