from utils import *

class Signal(np.ndarray):
    def __new__(cls, data, fs):
        if not isinstance(data, Signal):
            data = np.asarray(data).view(cls)
            cls.fs = fs
            return data

    @property
    def n_ch(self):
        if self.data.squeeze().ndim == 1:
            return 1
        else:
            return 2

    @property
    def length(self):
        return len(self)

    @property
    def duration(self):
        return len(self)/self.fs


class BatchProcessor:

    def __init__(self, fs):
        self.fs = fs

    def __call__(self, processor, audio_files, output_dir, out_extension="out"):
        for songpath in tqdm(audio_files):
            st = time.time()
            audio = load_audio_file(songpath, fs=self.fs)
            print(round(time.time()-st,3), os.path.basename(songpath))
            output = processor(audio)
            out_path = os.path.join(output_dir, os.path.basename(songpath)) + f'.{out_extension}'
            pickledump(output, out_path)



class Trim(object):
    """
    Trims audio to given length before processing
    if len==0: trim from 'start' to end of audio
    if start=='random', trim from a random time for len duration
    if start=='middle', trims audio from the middle symmetrically
    """
    def __init__(self, start=0, len=10, seed=0):
        self.start = start
        self.trim_len_sec = len
        self.seed = seed

    def __call__(self, signal):
        return trim(signal, self.start, self.trim_len_sec, self.seed)


class Normalize(object):
    """
    Normalizes wave domain audio to have maximum amplitude
    """
    def __init__(self):
        pass

    def __call__(self, signal):
        return madmom.audio.signal.normalize(signal)


class BPFilter(object):
    """
    Applies a Butterworth bandpass filter to the signal
    """
    def __init__(self, fl, fh, fs, order=2):
        self.coefs = self._filter_coef(lowcut=fl, highcut=fh, fs=fs, order=order)

    def __call__(self, signal):
        b, a = self.coefs[0], self.coefs[1]
        return np.asanyarray(lfilter(b, a, signal), dtype=signal.dtype)

    def _filter_coef(self, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        [b, a] = butter(order, [low, high], btype='band')
        return b, a


class Remix(object):
    """
    Remix signal to have desired number of channels
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def __call__(self, signal):
        return madmom.audio.signal.remix(signal, self.num_channels)


class TrimBoundarySilence(object):
    def __init__(self, where='fb'):
        self.where = where

    def __call__(self, signal):
        return madmom.audio.signal.trim(signal, self.where)

# -----------------------------------------------------------

# Functionals

def trim_boundary_silence(signal, where='fb'):
    return madmom.audio.signal.trim(signal, where)

def normalize(signal):
    return madmom.audio.signal.normalize(signal)

def trim(signal, start=0, len=10, seed=0):
    trim_len_sec = len
    if start in ['random']:
        rand_st = check_random_state(seed)
        first = rand_st.randint(0, signal.length - (trim_len_sec * signal.fs) - 1)
        trim_len_samples = int(trim_len_sec * signal.fs)
        last = first + trim_len_samples
        assert last < signal.length, \
            f"trim end point {last} exceeds audio length {signal.length}"
        trimmed_audio = signal[first:last]
        return trimmed_audio

    elif start in ['middle']:
        trim_len_samples = int(trim_len_sec * signal.fs)
        first = signal.length // 2 - trim_len_samples // 2
        last = first + trim_len_samples
        assert last < signal.length, \
            f"trim end point {last} exceeds audio length {signal.length}"
        trimmed_audio = signal[first:last]
        return trimmed_audio

    else:
        assert isinstance(start, float), "start needs to be 'random', 'middle', or a float"
        first = int(start * signal.fs)
        last = first + int(trim_len_sec * signal.fs)
        assert last < signal.length, \
            f"trim end point {last} exceeds audio length {signal.length}"
        trimmed_audio = signal[first:last]
        return trimmed_audio
