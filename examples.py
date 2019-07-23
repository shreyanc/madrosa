from utils import *
import temporal
import spectral

AUDIO_DIR = '/home/shreyan/PROJECTS/madrosa_extra/test_audio/'
EXAMPLE_AUDIO_PATH = os.path.join(AUDIO_DIR, '003.mp3')
SAMPLING_FREQ = 22050
OUTPUT_DIR = '/home/shreyan/PROJECTS/#synced/madrosa/'


def example_recon():
    # Initialize objects for audio processors
    normalize = temporal.Normalize()
    remove_silence = temporal.TrimBoundarySilence()
    extract_middle_part = temporal.Trim(start='middle')

    # Initialize a composite object that performs the operations
    ops = temporal.Compose([
        normalize,
        remove_silence,
        extract_middle_part
    ])

    # Load audio
    audio = load_audio_file(EXAMPLE_AUDIO_PATH, fs=SAMPLING_FREQ, mono=True)

    # Initialize a Signal object with the audio
    sig = temporal.Signal(audio, SAMPLING_FREQ)

    # Process the signal
    sig_processed = ops(sig)

    # Calculate the stft
    stft = spectral.calculate_stft(sig_processed)

    # Get the mel spectrogram from the stft
    mel_spec = stft.make_mel_spectrogram()

    # Get the filterbank
    fb = mel_spec.filterbank

    # Reconstruct
    import reconstruction
    reconstruction.reconstruct(mel_spec, fb, stft,
                               out=os.path.join(OUTPUT_DIR, 'recon_test.wav'))


if __name__=='__main__':
    example_recon()