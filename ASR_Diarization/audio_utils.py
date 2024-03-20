import resampy
import soundfile as sf


def conv_sr(audio_file_path, desired_sr=16000):
    audio, sr = sf.read(audio_file_path) #sampling rate aligned with whisper
    if sr != desired_sr:
        return resampy.resample(audio.T, sr, desired_sr).T #transposing audio to (len, channels) to fit resampy
    return audio.T

def split_wav(path):
    audio, sr = sf.read(path)
    return audio[:, 0], audio[:, 1]
