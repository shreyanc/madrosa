from utils import *

def reconstruct(modified_spec, fb, original_stft, out=None):
    mag = np.dot(modified_spec.T, fb).T
    new_stft = mag * original_stft / np.abs(original_stft)
    sig = librosa.istft(new_stft,
                        hop_length=original_stft.hop_length,
                        win_length=original_stft.win_length)
    if out is not None:
        librosa.output.write_wav(out, sig, original_stft.fs, norm=True)

    return sig