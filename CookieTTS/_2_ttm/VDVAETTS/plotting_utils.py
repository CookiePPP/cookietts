import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram, range=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, cmap='inferno', aspect="auto", origin="lower",
                   interpolation='none')
    if range is not None:
        assert len(range) == 2, 'range params should be a 2 element List of [Min, Max].'
        assert range[1] > range[0], 'Max (element 1) must be greater than Min (element 0).'
        im.set_clim(range[0], range[1])
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_time_series_to_numpy(target,# Numpy FloatTensor[T]
                                pred,# Numpy FloatTensor[T]
                          pred2=None,# Numpy FloatTensor[T]
                                xlabel="Frames (Green target, Red predicted)",
                                ylabel="Gate State",
                                alpha=0.5):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(target)), target, alpha=alpha,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(pred)), pred, alpha=alpha,
               color='red', marker='.', s=1, label='predicted')
    if pred2 is not None:
        ax.scatter(range(len(pred2)), pred2, alpha=alpha,
                    color='blue', marker='.', s=1, label='predicted')
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
