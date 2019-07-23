from utils import *


class TimeFreqRepresentation(np.ndarray):
    """
    The time-frequency class
    """
    def __new__(cls, stft, fs, hop_length=512, win_length=2048):
        if not isinstance(stft, TimeFreqRepresentation):
            stft = stft.view(cls)
            cls.fs = fs
            cls.hop_length = hop_length
            cls.win_length = win_length
            return stft

    @property
    def n_bins(self):
        return self.shape[0]

    @property
    def n_fft(self):
        return 2 * (self.n_bins - 1)

    @property
    def bin_freqs(self):
        return np.linspace(0, self.fs/2, self.n_bins)

    @property
    def complex_ph(self):
        return self/np.abs(self)

    @property
    def mag(self):
        return np.abs(self)

    def scaled_mag(self):
        return librosa.core.amplitude_to_db(self.mag)

    @property
    def phase(self):
        return  np.angle(self)

    @property
    def power_spectral_density(self):
        return np.power(self.mag, 2)

    @property
    def mel_spectrogram(self):
        return librosa.feature.melspectrogram(sr=self.fs, S=self.mag)

    def make_mel_spectrogram(self):
        melspec = self.mel_spectrogram
        return MelSpectrogram(melspec, self.fs, self.n_fft)


class MelSpectrogram(np.ndarray):
    def __new__(cls, melspec_mag, fs, n_fft):
        if not isinstance(melspec_mag, MelSpectrogram):
            melspec = melspec_mag.view(cls)
            cls.fs = fs
            cls.n_fft = n_fft
            return melspec

    @property
    def n_bins(self):
        return self.shape[0]

    @property
    def bin_freqs(self):
        return librosa.mel_frequencies(self.n_bins, fmax=self.fs/2)

    @property
    def scaled(self):
        return librosa.core.amplitude_to_db(self)

    @property
    def filterbank(self):
        return librosa.filters.mel(self.fs, self.n_fft)


def calculate_stft(signal, n_fft=2048, hop_length=512, win_length=2048):
    return TimeFreqRepresentation(librosa.stft(signal,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               win_length=win_length),
                                  signal.fs,
                                  hop_length=hop_length,
                                  win_length=win_length)

def calculate_melspec(signal, n_fft=2048, hop_length=512, win_length=2048):
    stft = calculate_stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return stft.make_mel_spectrogram()

