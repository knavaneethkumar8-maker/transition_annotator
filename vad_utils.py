import librosa
import numpy as np

SR=16000
NFFT=288
HOP=144
FRAMES=64


def extract_feature(y):

    if len(y)<NFFT:
        y=np.pad(y,(0,NFFT-len(y)))

    mel=librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=NFFT,
        hop_length=HOP,
        n_mels=64,
        center=False
    )

    mel=librosa.power_to_db(mel)

    if mel.shape[1]<FRAMES:
        mel=np.pad(mel,((0,0),(0,FRAMES-mel.shape[1])))

    mel=mel[:,:FRAMES]

    return mel.astype(np.float32)