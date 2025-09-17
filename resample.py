import librosa

def resample(y, og, new):
    y_resampled = librosa.resample(y=y, orig_sr=og, target_sr=new)
    return y_resampled
    