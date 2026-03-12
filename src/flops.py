import torch
from thop import profile

from src.model import SpeechCommandCNN


def compute_model_flops(n_mels=80, groups=1, input_length=16000):
    model = SpeechCommandCNN(n_mels=n_mels, groups=groups)
    model.eval()

    dummy_input = torch.randn(1, input_length)

    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs

    return {
        "macs": int(macs),
        "flops": int(flops),
        "params_thop": int(params),
    }