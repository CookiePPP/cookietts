import numpy as np
import torch
import soundfile as sf
import librosa
from scipy.io.wavfile import read

def load_wav_to_torch(full_path, target_sr=None, min_sr=None, remove_dc_offset=True, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        if return_empty_on_exception:
            print(ex)
            return [], sampling_rate or target_sr or 48000
        else:
            raise ex
    
    if min_sr is not None:
        if return_empty_on_exception and not (min_sr < sampling_rate):
            return [], sampling_rate or target_sr or 48000
        assert min_sr < sampling_rate, f'Expected sampling_rate greater than or equal to {min_sr:.0f}, got {sampling_rate:.0f}.\nPath = "{full_path}"'
    
    if len(data.shape) > 1: # if audio has more than 1 channels,
        data = data[:, 0]   # extract/use the first channel.
        assert len(data) > 2# Also check duration of audio file is > 2 samples (because otherwise the slice operation was probably on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# check for Nan/Inf in audio files
        return [], sampling_rate or target_sr or 48000
    assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found in audio file\n"{full_path}"'
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), sampling_rate, target_sr))
        if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
            return [], sampling_rate or target_sr or 48000
        assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found after resampling audio\n"{full_path}"'
        
        if remove_dc_offset:
            data = data - data.mean()
        abs_max = data.abs().max()
        if abs_max > 1.0:
            data /= abs_max
        sampling_rate = target_sr
        assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found after inf-norm rescaling audio\n"{full_path}"'
    
    return data, sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line_strip.split(split) for line_strip in (line.strip() for line in f) if line_strip and line_strip[0] is not ";"]
    return filepaths_and_text


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


@torch.jit.script
def DTW(batch_pred, batch_target, scale_factor: int, range_: int):
    """
    Calcuates ideal time-warp for each frame to minimize L1 Error from target.
    Params:
        scale_factor: Scale factor for linear interpolation.
                      Values greater than 1 allows blends neighbouring frames to be used.
        range_: Range around the target frame that predicted frames should be tested as possible candidates to output.
                If range is set to 1, then predicted frames with more than 0.5 distance cannot be used. (where 0.5 distance means blending the 2 frames together).
    """
    assert range_ % 2 == 1, 'range_ must be an odd integer.'
    assert batch_pred.shape == batch_target.shape, 'pred and target shapes do not match.'
    assert len(batch_pred) == 3, 'input Tensor must be 3d of dims [batch, height, time]'
    assert len(batch_target) == 3, 'input Tensor must be 3d of dims [batch, height, time]'
    
    batch_pred_dtw = batch_pred * 0.
    for i, (pred, target) in enumerate(zip(batch_pred, batch_target)):
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        
        # shift pred into all aligned forms that might produce improved L1
        pred_pad = torch.nn.functional.pad(pred, (range_//2, range_//2))
        pred_expanded = torch.nn.functional.interpolate(pred_pad, scale_factor=float(scale_factor), mode='linear', align_corners=False)# [B, C, T] -> [B, C, T*s]
        
        p_shape = pred.shape
        pred_list = []
        for j in range(scale_factor*range_):
            pred_list.append(pred_expanded[:,:,j::scale_factor][:,:,:p_shape[2]])
        
        pred_dtw = pred.clone()
        for pred_interpolated in pred_list:
            new_l1 = torch.nn.functional.l1_loss(pred_interpolated, target, reduction='none').sum(dim=1, keepdim=True)
            old_l1 = torch.nn.functional.l1_loss(pred_dtw, target, reduction='none').sum(dim=1, keepdim=True)
            pred_dtw = torch.where(new_l1 < old_l1, pred_interpolated, pred_dtw)
        batch_pred_dtw[i:i+1] = pred_dtw
    return batch_pred_dtw
