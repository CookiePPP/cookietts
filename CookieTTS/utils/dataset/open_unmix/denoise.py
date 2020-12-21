import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
from . import model
from . import utils
import warnings
import tqdm
from contextlib import redirect_stderr
import io
from CookieTTS.utils.dataset.utils import load_wav_to_torch

def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            state['sample_rate'],
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model.OpenUnmix(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )

        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)
        return unmix


def istft(spect, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        spect / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio

models_global = [load_model(target=target, model_name='umxhq', device='cpu',) for target in ['vocals', 'other', 'drums', 'bass']]

def separate_vocals(
    audiopath,
    niter=1, softmask=False, alpha=1.0,
    device='cpu',
):
    """
    Performing the separation on audio input
    Parameters
    ----------
    audiopath: str
         path to audio file `.wav` or `.flac` or `.ogg`
    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.
    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False
    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0
    device: str
        set torch device. Defaults to `cpu`.
    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.
    """
    models = [model.to(device) for model in models_global]
    # convert numpy audio to torch
    audio_torch = load_wav_to_torch(audiopath, target_sr=44100)[0][None, None, :].to(device)
    audio_torch /= audio_torch.abs().max()
    audio_torch = audio_torch.repeat(1, 2, 1)
    
    filters = []
    
    for model in models:
        denoise_filter = model(audio_torch).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            denoise_filter = denoise_filter**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        filters.append(denoise_filter[:, 0, ...])# remove sample dim
    
    filters = np.transpose(np.array(filters), (1, 3, 2, 0))
    
    spect = model.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    spect = spect[..., 0] + spect[..., 1]*1j
    spect = spect[0].transpose(2, 1, 0)
    
    Y = norbert.wiener(filters, spect.astype(np.complex128), niter,
                       use_softmask=softmask)
    
    denoised_audio = istft(
        Y[..., 0].T,
        n_fft=model.stft.n_fft,
        n_hopsize=model.stft.n_hop
    )
    
    return torch.from_numpy(denoised_audio)[0].to(device)

def get_non_voiced_avglogmag(audiopath):
    
    return logmag