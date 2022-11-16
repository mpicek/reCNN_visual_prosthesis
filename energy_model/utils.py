import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import math
import scipy.signal as signal


def create_grating(sf, ori, phase, wave, res, xlim, ylim):
    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :return: numpy array of shape (imsize, imsize)
    """
    # Get x and y coordinates
    X = np.linspace(xlim[0], xlim[1], res[0])
    Y = np.linspace(ylim[0], ylim[1], res[1])
    x, y = np.meshgrid(X, Y)

    # Get the appropriate gradient
    gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y

    # Plug gradient into wave function
    if wave is 'sin':
        grating = np.sin((2 * math.pi * gradient) * sf + (phase * math.pi) / 180)
    elif wave is 'sqr':
        grating = signal.square((2 * math.pi * gradient) / sf + (phase * math.pi) / 180)
    else:
        raise NotImplementedError

    return grating


def plot_f(f, title="", vmin=None, vmax=None, return_plt=False, ticks=True, cmap=None, graph_size=12):
    """Luca's simple function for plotting of the Gabor Filter

    Args:
        f (_type_): _description_
        title (str, optional): _description_. Defaults to "".
        vmin (_type_, optional): _description_. Defaults to None.
        vmax (_type_, optional): _description_. Defaults to None.
        return_plt (bool, optional): _description_. Defaults to False.
        ticks (bool, optional): _description_. Defaults to True.
        cmap (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    plt.clf()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gcf().set_size_inches(graph_size, graph_size)
    if type(f) == torch.Tensor:
        f = f.detach().cpu().numpy().squeeze()
    m = np.max(np.abs(f))
    if vmin is None:
        min = -m
    else:
        min = vmin
    if vmax is None:
        max = m
    else:
        max = vmax
    if cmap == "greys":
        color_map = "Greys_r"
    else:
        color_map = cm.coolwarm
    plt.imshow(f, vmax=max, vmin=min, cmap=color_map)
    plt.title(title)
    plt.colorbar()
    if ticks == False:
        plt.xticks([])
        plt.yticks([])
    if return_plt:
        return plt
    else:
        plt.show()