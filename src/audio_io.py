import soundfile as sf
import torch
import torchaudio


def patch_torchaudio():
    def safe_load(path, *args, **kwargs):
        data, sr = sf.read(path, dtype="float32")

        if data.ndim == 1:
            waveform = torch.tensor(data).unsqueeze(0)
        else:
            waveform = torch.tensor(data.T)

        return waveform, sr

    torchaudio.load = safe_load