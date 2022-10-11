import itertools
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def plot_inertial_gyroscope_multiple(title, y_label, legends, data, save=False, show_figure=True):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    for i in range(len(data)):
        ax.plot(data[i], label=legends[i])
    ax.set(title=title, ylabel=y_label, xlabel='Timesteps')
    fig.legend()
    if save:
        _save_plot(fig, '%s.png' % title.strip())
    if show_figure:
        plt.show()

def plot_inertial(data, title, y_label, save=False, show_figure=True):
    """
    Plots inertial data
    :param data: ndarray
    :param title: Plot title
    :param y_label: Y axis label
    :param save: Save figure to file
    :param show_figure: Show figure
    """
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots()
    ax.plot(data[:, 0], label='X')
    ax.plot(data[:, 1], label='Y')
    ax.plot(data[:, 2], label='Z')
    ax.set(title=title, ylabel=y_label, xlabel='Timesteps')
    fig.legend()
    if save:
        _save_plot(fig, '%s.png' % title.strip())
    if show_figure:
        plt.show()

def plot_inertial_subplots(top_data, bottom_data, top_title, bottom_title, y_label, save=False, show_figure=True):
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(top_data[:, 0], label='X')
    ax1.plot(top_data[:, 1], label='Y')
    ax1.plot(top_data[:, 2], label='Z')
    ax1.set(title=top_title, ylabel=y_label)

    ax2.plot(bottom_data[:, 0], label='X')
    ax2.plot(bottom_data[:, 1], label='Y')
    ax2.plot(bottom_data[:, 2], label='Z')
    ax2.set(title=bottom_title, ylabel=y_label, xlabel='Timesteps')

    fig.legend()
    if save:
        _save_plot(fig, '%s-%s.png' % (top_title.strip(), bottom_title.strip()))
    if show_figure:
        plt.show()


def _save_plot(fig, filename):
    """
    Saves a plot to the 'plots' directory in the project root folder
    :param fig: figure
    :param filename: filename with extension
    :return:
    """
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots'))
    datetime = time.strftime("%Y%m%d_%H%M", time.localtime())
    filename = datetime + '_' + filename
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    final_filename = os.path.join(save_dir, filename)
    fig.savefig(final_filename)
    print('Saved figure in ' + final_filename)
