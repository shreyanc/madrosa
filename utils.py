# All imports for the whole module are located here.
# Just do 'from utils import *' to import these and the helper functions

import os
import librosa
import madmom
import time
import pickle
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, lfilter
from sklearn.utils import check_random_state
from matplotlib import pyplot as plt


# Helper functions --------------------------

def pickledump(source, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(source, f)


def directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def load_audio_file(filepath, fs, mono=True, mode='normal'):
    """
    Loads the audiofile from disk as a numpy array using subprocess call to ffmpeg
    :param filepath: path of the audio file
    :return: numpy array
    """
    # TODO: mode is temporary, remove after validating if load function of librosa 0.7 is faster or not
    if mode in ['normal']:
        import subprocess
        ch = 1 if mono else 2
        call = ['ffmpeg', '-v', 'quiet', '-y', '-i', filepath, '-f', 's16le', '-ac', f'{ch}', '-ar', f'{fs}', 'pipe:1']
        proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=-1)
        signal, _ = proc.communicate()
        signal = np.frombuffer(signal, dtype=np.int16).astype(np.float32)
        if not mono:
            # TODO: implement stereo
            signal = signal.reshape((-1, ch)).T
        return signal
    elif mode in ['librosa']:
        signal = librosa.core.load(path=filepath, sr=fs, mono=mono)[0]
        return signal
    else:
        raise NotImplementedError


def get_songfilepaths(dirpath):
    """
    Returns full paths of all wav and mp3 files in a directory, including subdirectories
    :param dirpath: path of root directory
    :return: list of paths
    """
    song_filepaths = []
    for dirpath, _, filenames in os.walk(os.path.join(dirpath, '')):
        for songname in filenames:
            if songname.split('.')[-1] in ['wav', 'mp3']:
                song_filepaths.append(os.path.join(dirpath, songname))
    return song_filepaths

# --------------------------------------------

class Compose(object):
    """
    Applies a pipeline of transforms.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sig):
        for t in self.transforms:
            sig = t(sig)
        return sig

