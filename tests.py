from utils import *
import temporal
import spectral
from librosa import display as d

AUDIO_DIR = '/home/shreyan/PROJECTS/madrosa_extra/test_audio/'
EXAMPLE_AUDIO_PATH = os.path.join(AUDIO_DIR, '7401.mp3')
FS = 22050

def test_batch_process():
    normalize = temporal.Normalize()
    mono = temporal.Remix(num_channels=1)
    trim = temporal.Trim(start='middle')
    operations = temporal.Compose([
        trim,
        normalize,
        mono
    ])
    filepaths = temporal.get_songfilepaths(AUDIO_DIR)
    process_all_files = temporal.BatchProcessor(fs=FS)

    process_all_files(operations, filepaths, directory(os.path.join(AUDIO_DIR, 'output')), out_extension='stft')


def test_transform(transform):
    sig = temporal.load_audio_file(os.path.join(AUDIO_DIR, '7401.mp3'))
    out = transform(sig)
    librosa.output.write_wav('./transformed_out.wav', out, 22050, norm=True)
    return out


def test_signal_class():
    sig = load_audio_file(EXAMPLE_AUDIO_PATH, fs=22050, mono=True)
    signal = temporal.Signal(sig, 22050)
    pass


def test_spec():
    norm = temporal.Normalize()
    trim = temporal.TrimBoundarySilence()
    ops = temporal.Compose([
        norm,
        trim
    ])
    sig = temporal.Signal(load_audio_file(EXAMPLE_AUDIO_PATH, fs=22050, mono=True), 22050)
    # s = sig[5:10]
    sig_processed = ops(sig)
    mel = spectral.calculate_melspec(sig_processed)
    pass


def test_recon():
    norm = temporal.Normalize()
    trim = temporal.TrimBoundarySilence()
    trim_mid = temporal.Trim(start='middle')
    ops = temporal.Compose([
        norm,
        trim,
        trim_mid
    ])
    sig = temporal.Signal(load_audio_file(EXAMPLE_AUDIO_PATH, fs=FS, mono=True), FS)
    sig_processed = ops(sig)

    librosa.output.write_wav('/home/shreyan/PROJECTS/#synced/madrosa/recon_test.wav',
                             sig_processed,
                             FS, norm=True)

    stft = spectral.calculate_stft(sig_processed)

    import reconstruction
    reconstruction.reconstruct(stft.mel_spectrogram, librosa.filters.mel(stft.fs, stft.n_fft), stft,
                               out='/home/shreyan/PROJECTS/#synced/madrosa/recon_test.wav')
    pass


if __name__=='__main__':
    # normalize = processors.Normalize()
    # filter = processors.BPFilter(20, 8000, 22050)
    # trans = processors.Compose([
    #     normalize,
    #     filter
    # ])
    # out = test_transform(trans)
    #
    # test_batch_process()
    # test_signal_class()
    # test_spec()
    test_recon()
