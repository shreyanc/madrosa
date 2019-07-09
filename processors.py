import os
import librosa
import pickle
import numpy as np
from tqdm import tqdm


# Helper functions --------------------------

def pickledump(source, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(source, f)

# --------------------------------------------


def load_audio_file(filepath):
    """
    Loads the audiofile from disk as a numpy array using subprocess call to ffmpeg
    :param filepath: path of the audio file
    :return: numpy array
    """
    import subprocess
    call = ['ffmpeg', '-v', 'quiet', '-y', '-i', filepath, '-f', 's16le', '-ac', '1', '-ar', '22050', 'pipe:1']
    proc = subprocess.Popen(call, stdout=subprocess.PIPE, bufsize=-1)
    signal, _ = proc.communicate()
    signal = np.frombuffer(signal, dtype=np.int16).astype(np.float32)
    return signal


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


class SpectrogramProcessor:

    def __init__(self, frame_size=2048):
        self.frame_size = frame_size

    def batch_extract_magnitude_spectrogram(self, in_dirpath, out_dirpath, amp2db=True):
        song_filepaths = get_songfilepaths(in_dirpath)

        for songpath in tqdm(song_filepaths):
            _, mag_spec, _ = self._compute_spectrogram(songpath, amp2db=amp2db)
            out_path = os.path.join(out_dirpath, os.path.basename(songpath)) + '.mag'
            pickledump(mag_spec, out_path)

    def batch_extract_phase_spectrogram(self, in_dirpath, out_dirpath, mode='complex'):
        song_filepaths = get_songfilepaths(in_dirpath)

        for songpath in tqdm(song_filepaths):
            _, _, ph_spec = self._compute_spectrogram(songpath, mode=mode)
            out_path = os.path.join(out_dirpath, os.path.basename(songpath)) + '.ph'
            pickledump(ph_spec, out_path)

    def batch_extract_stft(self, in_dirpath, out_dirpath, amp2db=True, mode='complex'):
        song_filepaths = get_songfilepaths(in_dirpath)

        for songpath in tqdm(song_filepaths):
            stft, _, _ = self._compute_spectrogram(songpath, amp2db=amp2db, mode=mode)
            out_path = os.path.join(out_dirpath, os.path.basename(songpath)) + '.stft'
            pickledump(stft, out_path)

    @staticmethod
    def _compute_spectrogram(filepath, amp2db=True, mode='complex'):
        """
        Loads audio, computes the spectrogram and returns it  along with magnitude and phase terms such that
        D = D_mag * D_ph, where D is the complex spectrogram
        :param filepath: filepath of audio
        :param amp2db: return D_mag as db or not
        :param mode: return D_ph as 'complex' or 'radians'
        :return: magnitude and phase components of spectrogram
        """
        y = load_audio_file(filepath)
        D = librosa.stft(y)
        D_mag, D_ph = librosa.magphase(D)
        if amp2db:
            D_mag = librosa.amplitude_to_db(D_mag, ref=np.max)
        if mode in ['radians']:
            D_ph = np.angle(D_ph)
        return D, D_mag, D_ph


