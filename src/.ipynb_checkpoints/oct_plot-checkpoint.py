'Plotting functions & data visualization'
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
import seaborn as sns

from src.compute import *

from scipy.optimize import curve_fit


def plot_quality_distribution (df,  output_file, fig_name=None, bins_=100, log=False):
    """Based on the POC score, plot OCTs quality distribution.
    Args:
        df (DataFrame) : contains the scores of each OCTs
        output_file (str) :  name of the directory to save the plot
        fig_name(str) : name of the figure
        bins_ (int) :  number of bins to plot the histogram.

    """
    #Histogram
    
    palette=sns.color_palette("crest",n_colors=150)
    sns_plot=sns.histplot(df, x="score", bins = bins_, color=palette[0])
    sns.despine()
    fig=sns_plot.get_figure()

    # Plot config
    plt.title('Distribution of quality score')
    plt.xlabel(f'Quality score with a mean of {np.round(np.mean(df.score),2)} ')
    plt.ylabel('OCT Count')
    if(log): plt.yscale('log')
    
    plt.show()
    

    # Saving
    fig_file=output_file + f"/{fig_name}.png"
    fig.savefig(fig_file, transparent=True)



def plot_Bscans(volume, frames):
    """Produces a figure composed of the list of frames

    Args:
        volume (np.ndarray): OCT cube loaded as a numpy array
        frames (list[int]): list of frames we want to plot

    Returns:
        matplotlib.Figure: figure of all frames
    """

    assert len(volume.shape) == 3, " Provided array has the wrong number of dimensions"
    assert frames, " No frames provided"

    # Discard invalid frames indeces
    frames = [f for f in frames if (f < len(volume) and f >= 0)]
    # Define figure
    fig, ax = plt.subplots(1, len(frames), figsize=(20, 20))
    # Configure subplots proportions
    fig.subplots_adjust(
        top=1,
        bottom=0,
        left=0,
        right=0.2 *
        len(frames),
        wspace=0.05,
        hspace=0)

    for id, frame in enumerate(frames):
        if (len(frames) == 1):
            ax[0].imshow(volume[frame, :, :], cmap='gray')
            title = 'img ' + str(frame)
            ax[0].set_title(title)
            ax[0].axis('off')
        else:
            ax[id].imshow(volume[frame, :, :], cmap='gray')
            title = 'img ' + str(frame)
            ax[id].set_title(title)
            ax[id].axis('off')
    return fig


def plot_Bscans_from_path(oct_path, frames):
    """Load image from path and plot B-scans list
    Args:
        oct_path (str): path to the OCT scan folder
        frames (list[int]): list of frames we want to plot

    Returns:
        matplotlib.Figure : figure of all frames
    """
    oct = load_array_from_folder(oct_path)
    return plot_Bscans(oct, frames)


def plot_fundus(volume: np.ndarray, aspect=1.0, title = None):
    """Produces a figure the fundus by computing the mean of each A-scans given an image

    Args:
        volume (np.ndarray): OCT cube as a 3D numpy array
        aspect (double): aspect ratio of the image to plot. Defaults to 1.9
        title (str) : title of figure
    Returns:
       matplotlib.Figure : 2D image of fundus
    """
    assert len(
        volume.shape) == 3, " Provided array has the wrong number of dimensions"

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # computes the fundus and show the resulting by doing the mean along the y
    # axis, axis = 1 in {0, 1, 2}
    ax.imshow(np.mean(volume, axis=1), aspect=aspect, cmap='gray')
    ax.axis('off')
    if title is None : 
        ax.set_title("Fundus reconstruction from OCT")
    else :
        ax.set_title(title)

    return fig


def plot_fundus_from_path(oct_path: str, aspect = 1.0, title = None):
    """Load an OCT image and produces a figure the fundus by computing the mean of each A-scans given an image

    Args:
        oct_path (str): path to the OCT scan folder
        aspect (double): aspect of fundus reconstruction
        title (str) : title of figure
    Returns:
        plt.figure: 2D image of fundus
    """

    oct = np.array(load_array_from_folder(oct_path))

    return plot_fundus(oct, aspect, title)


def plot_normalization_comp(volume: np.ndarray, frame: int):
    """Produces a figure plotting a B-scan and its normalized version for comparison

    Args:
        volume (np.ndarray): oct cube, ndarray of OCT B-scans
        frame (int): frame to plot

    Returns:
        plt.figure: return a figure corresponding to loaded frame
    """

    assert len(volume.shape) == 3, " Provided array has the wrong number of dimensions"
    assert (frame >= 0 and frame < volume.shape[0]), " frame to plot out of range"

    volume_normd = a_scan_normalization(volume)

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    # Adjust parameters for each number of frames
    fig.subplots_adjust(
        top=1,
        bottom=0,
        left=0,
        right=0.5,
        wspace=0.1,
        hspace=0.1)
    ax[0].imshow(volume[frame, :, :], cmap='gray')
    ax[1].imshow(volume_normd[frame, :, :], cmap='gray')
    title = 'img ' + str(frame)
    title_normd = title + ' normalized'
    ax[0].set_title(title)
    ax[1].set_title(title_normd)
    ax[0].axis('off')
    ax[1].axis('off')
    return fig


def plot_superposition_frame(original_volume: np.ndarray,
                             approximated_volume: np.ndarray, frame: int):
    """Produces a figure of the superposition of a B-scan of volume A and a B-scan of volume B.
    B-scan A is shown on the left and is represented in the red channel in center image.
    B-scan B is shown on the right and is represented in the green channel in center image.

    Args:
        original_volume (np.ndarray): volume to be represented on the left and red channel
        approximated_volume (np.ndarray): volume to be represented on the right and green channel
        frame (int): index of B-scan to plot

    Returns:
        plt.figure: figure of the three images, B-scan A, superposition, B-scan B
    """

    assert len(original_volume.shape) == 3, " Provided array has the wrong number of dimensions"
    assert len(approximated_volume.shape) == 3, " Provided array has the wrong number of dimensions"
    assert (frame >= 0 and frame < original_volume.shape[0]), " frame to plot out of range, for original_volume"
    assert (frame >= 0 and frame < approximated_volume.shape[0]), " frame to plot out of range, for approximated_volume"

    figure, axis = plt.subplots(1, 1, figsize=(15, 15))

    red_channel = np.concatenate([original_volume[frame], original_volume[frame], approximated_volume[frame]], axis=1)
    blue_channel = np.concatenate([original_volume[frame], approximated_volume[frame], approximated_volume[frame]], axis=1)
    green_channel = np.concatenate([original_volume[frame], np.zeros(original_volume[frame].shape), approximated_volume[frame]], axis=1)

    rgb_image = np.array([red_channel, blue_channel, green_channel]).T
    rgb_image = np.rot90(rgb_image)
    rgb_image = rgb_image[::-1, :, :]

    axis.text(rgb_image.shape[1] / 6 - 50, -10,
              'original image', fontsize=12, color='k')
    axis.text(rgb_image.shape[1] / 3 + 75, -10,
              'superposition', fontsize=12, color='k')
    axis.text(rgb_image.shape[1] * 2 / 3 + 75, -10,
              'neighbour approximation', fontsize=12, color='k')

    axis.imshow(rgb_image)
    axis.axis('off')

    axis.imshow(rgb_image)
    axis.axis('off')

    return figure


def plot_superposition(original_img: np.ndarray,
                       approximated_img: np.ndarray):
    """Produces a figure of the superposition of a B-scan of volume A and a B-scan of volume B.
    B-scan A is shown on the left and is represented in the red channel in center image.
    B-scan B is shown on the right and is represented in the green channel in center image.

    Args:
        original_img (np.ndarray): image to be represented on the left and red channel
        approximated_img (np.ndarray): image to be represented on the right and green channel

    Returns:
        matplotlib.Figure : superposed volume
    """

    assert len(original_img.shape) == 2, " Provided array has the wrong number of dimensions"
    assert len(approximated_img.shape) == 2, " Provided array has the wrong number of dimensions"
    assert original_img.shape == approximated_img.shape, "Images not of the same dimensions"
    
    figure, axis = plt.subplots(1, 1, figsize=(15, 15))

    red_channel = np.concatenate(
        [original_img, original_img, approximated_img], axis=1)
    blue_channel = np.concatenate(
        [original_img, approximated_img, approximated_img], axis=1)
    green_channel = np.concatenate([original_img, np.zeros(
        original_img.shape), approximated_img], axis=1)

    rgb_image = np.array([red_channel, blue_channel, green_channel]).T
    rgb_image = np.rot90(rgb_image)
    rgb_image = rgb_image[::-1, :, :]

    axis.text(rgb_image.shape[1] / 6 - 50, -10,
              'original image', fontsize=12, color='k')
    axis.text(rgb_image.shape[1] / 3 + 75, -10,
              'superposition', fontsize=12, color='k')
    axis.text(rgb_image.shape[1] * 2 / 3 + 75, -10,
              'neighbour approximation', fontsize=12, color='k')

    axis.imshow(rgb_image)
    axis.axis('off')

    axis.imshow(rgb_image)
    axis.axis('off')

    return figure


def fundus_along_score(volume: np.ndarray, list_of_scores: list, labels: list, colors=[
                       'blue', 'orange', 'green', 'red'], width=512, height=512):
    """Plot fundus reconstruction alongside multiple user specified score curves to better see discontuity-score impact

    Args:
        volume (np.ndarray): OCT data to analyze, in the form of a 3D numpy array.
        list_of_scores (list): list of scores to plot alongside fundus. In order to be lisible, the score has to be contained within [0, 1].
        labels (list): label to identify each score.
        colors (list, optional): colours to plot each score in. Defaults to ['blue', 'orange', 'green', 'red'].
        width (int, optional): desired width of fundus reconstruction. Defaults to 512.
        height (int, optional): desired height of fundus reconstruction. Defaults to 512.

    Returns:
        plt.figure: matplotlib figure
    """
    assert len(
        volume.shape) == 3, "Provided array has the wrong number of dimensions"

    assert len(labels) >= len(list_of_scores), "Not enough labels provided"

    assert len(colors) >= len(list_of_scores), "Not enough colours provided"

    fundus = cv2.resize(volume.mean(axis=1), (width, height))
    height, width = fundus.shape[0], fundus.shape[1] * 2

    figure, axis = plt.subplots(1, 1, figsize=(15, 15))

    axis.imshow(fundus, cmap='gray')

    # Ticks for grid & ylabels
    if volume.shape[0] == 128:
        yticks = 2 + 1
    if volume.shape[0] == 256:
        yticks = 4 + 1
    if volume.shape[0] == 320:
        yticks = 8 + 1
    else:
        yticks = 2 + 1

    # Horizontal grid
    axis.vlines(width / 2 + 5, ymin=0, ymax=height, color='k')
    for i, x in enumerate(np.linspace(width / 2 + 5, width, 11)):
        axis.vlines(x, ymin=0, ymax=height, color='k', alpha=0.3)
        axis.text(x - 10, -10, str(i / 10))

    # Vertical grid
    for y in np.linspace(0, height, yticks):
        axis.hlines(y, xmin=width / 2 + 5, xmax=width, color='k', alpha=0.3)

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)

    axis.text(width * 3 / 4, height + 15, 'Score')

    # Plot score curves
    for score, label, color in zip(list_of_scores, labels, colors):
        # Squish indeces to fit in figure
        n = volume.shape[0] - len(score)
        axis.plot([(i * width / 2) + width / 2 + 5 for i in score],
                  [(i + n) * (height / (len(score) + 2 * n))
                   for i in range(len(score))],
                  color=color,
                  label=label)

    # Plot y axis by warpping it to the correct dimensions for easier
    # readability

    plt.yticks(
        np.linspace(
            0, height, yticks), labels=np.linspace(
            0, volume.shape[0], yticks, dtype=int))
    axis.get_xaxis().set_visible(False)
    plt.legend()
    return figure

def display_fundus(volume: np.ndarray, width=512, height=512):
    assert len(volume.shape) == 3, "Provided array has the wrong number of dimensions"

    fundus = cv2.resize(volume.mean(axis=1), (width, height))
    height, width = fundus.shape[0], fundus.shape[1] * 2

    figure, axis = plt.subplots(1, 1, figsize=(10, 10))
    axis.imshow(fundus, cmap='gray')

    # Ticks for grid & ylabels
    if volume.shape[0] == 128:
        yticks = 2 + 1
    if volume.shape[0] == 256:
        yticks = 4 + 1
    if volume.shape[0] == 320:
        yticks = 8 + 1
    else:
        yticks = 2 + 1

    # Horizontal grid
    axis.vlines(width / 2 + 5, ymin=0, ymax=height, color='k')
    for i, x in enumerate(np.linspace(width / 2 + 5, width, 11)):
        axis.vlines(x, ymin=0, ymax=height, color='k', alpha=0.3)

    # Vertical grid
    for y in np.linspace(0, height, yticks):
        axis.hlines(y, xmin=width / 2 + 5, xmax=width, color='k', alpha=0.3)

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    # Title of x axis 
    axis.text(width * 3 / 4, height + 15, 'Image alignment (in pixels)')

    # Plot y axis by warpping it to the correct dimensions for easier readability
    plt.yticks(
        np.linspace(
            0, height, yticks), labels=np.linspace(
            0, volume.shape[0], yticks, dtype=int))
    axis.get_xaxis().set_visible(False)

    plt.legend(loc='upper center')
    return figure

def compute_fundus(volume, width=512, height=512):
    fundus = cv2.resize(volume.mean(axis=1), (width, height))
  
    return fundus

def fundus_along_POC(fundus, compound, label, x_max ,peak_list = [], title = None, color="slategrey", show_curve = True):
    """Plot fundus reconstruction alongside multiple user specified score curves to better see discontuity-score impact

    Args:
        volume (np.ndarray): OCT data to analyze, in the form of a 3D numpy array.
        compound (list): compound measure to plot alongside fundus. In order to be lisible, the score has to be contained within [0, x_max].
        label (str): label to identify score.
        x_max (float) : maximum of x axis for plotting compound. Will scale compound accordingly for representation
        peak_list (list, optional) : list of indeces of peaks to plot hlines. Defaults to empty list.
        color (str, optional): colours to plot score. Defaults to 'blue'.
        width (int, optional): desired width of fundus reconstruction. Defaults to 512.
        height (int, optional): desired height of fundus reconstruction. Defaults to 512.

    Returns:
        plt.figure: matplotlib figure
    """
  
    assert len(
        fundus.shape) == 2, "Provided array has the wrong number of dimensions"

    height, width = fundus.shape[0], fundus.shape[1] * 2

    figure, axis = plt.subplots(1, 1, figsize=(15, 15))
    size = compound[compound!=-9999.0].shape[0] + 1

    axis.imshow(fundus, cmap='gray')

    # Ticks for grid & ylabels
    if size == 128:
        yticks = 2 + 1
    if size == 256:
        yticks = 4 + 1
    if size == 320:
        yticks = 8 + 1
    else:
        yticks = 2 + 1

    # Horizontal grid
    axis.vlines(width / 2 + 5, ymin=0, ymax=height, color='k')
    xticks = np.linspace(0, x_max, 11)
    for i, x in enumerate(np.linspace(width / 2 + 5, width, 11)):
        axis.vlines(x, ymin=0, ymax=height, color='k', alpha=0.3)
        # x tick 
        xtick = str(xticks[i])
        # show xtick with no more than 4 digits
        axis.text(x - 10, -10, xtick[:min(len(xtick), 5)])

    # Vertical grid
    for y in np.linspace(0, height, yticks):
        axis.hlines(y, xmin=width / 2 + 5, xmax=width, color='k', alpha=0.3)

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    # Title of x axis 
    # axis.text(width * 3 / 4, height + 15, 'Image alignment (in pixels)')

    # Plot score curve
    n = size - len(compound)
    # Plot scaled compound value
    if show_curve:
        axis.plot([(i * (width / 2) * (1/x_max)) + width / 2 + 5 for i in compound],[(i + n) * (height / (len(compound) + 2 * n)) 
            for i in range(len(compound))],color=color,label=label)

    # Plot y axis by warpping it to the correct dimensions for easier readability
    plt.yticks(
        np.linspace(
            0, height, yticks), labels=np.linspace(
            0, size, yticks, dtype=int))
    axis.get_xaxis().set_visible(False)

    # Plot peak list
    if len(peak_list) > 0:
        # TODO : artificial shift of 1 unit for hlines to properly align peaks with POC score, investigate why
        # probably due to the length the score is plotted to (128, 256, 320) compared to score length (127, 255, 319)
        plt.hlines(np.array([peak_list]) * (height / size) + 1, 
            xmin=-100, xmax=1500, color='seagreen', alpha=1, label='detected_peaks', linestyles='dotted')
        # for peak in peak_list :
           #  plt.text(-20.0 , peak * (height / size) + 1.5, str(peak), fontsize=6, color='red')
           #  plt.text(width + 20.0 , peak * (height / size) + 1.5, str(peak), fontsize=6, color='red')
    # if not title is None:
       #  plt.title(title, loc='left')
    if show_curve:
        plt.legend(loc="upper right")

def generate_gif(volume: np.ndarray, axis: int, aspect=0.0, start=0,
                 end=0, step=1, name='mygif.gif'):
    """Generate Gif sliceshow of OCT cube in chosen axis and stepping

    Args:
        volume (np.ndarray): OCT cube as numpy array
        axis (int): axis in which to go through OCT. x, y or z.
        start (int, optional): start index. Defaults to 0, first slice.
        end (int, optional): end index. Defaults to last slice.
        step (int, optional): stepping. Defaults to 1.
        name (str, optional): Name of generated gif. Defaults to 'mygif.gif'.
    """
    # Check
    assert len(
        volume.shape) == 3, " Provided array has the wrong number of dimensions"

    if end <= start:
        # Default value of end
        end = volume.shape[axis]

    if not name.endswith('.gif') : name += '.gif'
    
    # Create empty temporary folder for storing images
    filenames = list()
    tmp_folder = 'tmp/'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.mkdir(tmp_folder)


    if axis == 0:
        volume = volume[start:end:step, :, :]
    if axis == 1:
        volume = volume[:, start:end:step, :]
    if axis == 2:
        volume = volume[:, :, start:end:step]

    for id in range(volume.shape[axis]):
        # plot the line chart
        plt.figure(figsize=(20, 20))
        if axis == 0:
            if aspect == 0.0 : aspect = 0.3
            plt.imshow((volume[id, :, :]), aspect=aspect)
        if axis == 1:
            if aspect == 0.0 : aspect = 1.0
            plt.imshow(np.transpose(volume[:, id, :]), aspect=aspect)
        if axis == 2:
            if aspect == 0.0 : aspect = 0.3
            plt.imshow(np.transpose(volume[:, :, id]), aspect=aspect)
        plt.axis('off')

        plt.title(str("Image " + str(id * step + start)), fontsize=30)
        # create file name and append it to a list
        filename = f'{id}.png'
        filenames.append(tmp_folder + filename)

        # save frame
        plt.savefig(tmp_folder + filename)
        plt.close()  # build gif
    with imageio.get_writer(name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    shutil.rmtree(tmp_folder)

def generate_gif_v2(volume: np.ndarray, axis: int, width=512, height=512, fps=1, name='mygif.gif'):
    """
    Generate Gif sliceshow of OCT cube in chosen axis and stepping
    Args:
    volume (np.ndarray): OCT cube as numpy array
    axis (int): axis in which to go through OCT. x, y or z.
    start (int, optional): start index. Defaults to 0, first slice.
    end (int, optional): end index. Defaults to last slice.
    step (int, optional): stepping. Defaults to 1.
    name (str, optional): Name of generated gif. Defaults to 'mygif.gif'.
    """
    # Check
    assert len(volume.shape) == 3, " Provided array has the wrong number of dimensions"

    if not name.endswith('.gif') : name += '.gif'

    image_list = [image.astype(np.uint8) for image in volume]

    if width is not None or height is not None:
        resized_images = []
        for image in image_list:
            resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            resized_images.append(resized_image)
        image_list = resized_images

    imageio.mimsave(name, image_list, fps=fps)




def plot_funduses(path_list: list, patient_id_list: list, aspect=1):
    """Plot multiple fundus recontsruction from list of paths and list of patient id

    Args:
        path_list (list): list of paths as str
        patient_id_list (list): list of patient id
        aspect (int, optional): ratio of fundus reconstruction, can be used to adat image shape into a square. Defaults to 1.

    Returns:
        matplotlib.Figure : fundus reconstruction with patient_id above them
    """
    # Define figure placement
    rows = 2
    columns = len(path_list) // 2 + len(path_list) % 2
    fig, ax = plt.subplots(
        rows, columns, figsize=(
            10, 3), constrained_layout=True)

    assert len(patient_id_list) >= len(
        path_list), " Not enough patient ids provided"
    for index, path in enumerate(path_list):

        volume = load_array_from_folder(path)

        assert len(
            volume.shape) == 3, " Provided array has the wrong number of dimensions"

        # computes the fundus and show the resulting by doing the mean along the y
        # axis, axis = 1 in {0, 1, 2}
        if index < columns:
            i = 0
            j = index
        else:
            i = 1
            j = index - columns

        ax[i][j].imshow(np.mean(volume, axis=1), aspect=aspect, cmap='gray')
        ax[i][j].axis('off')
        ax[i][j].set_title("pID " + str(patient_id_list[index]), fontsize=6)

    if columns > len(path_list) // 2:
        ax[rows - 1][columns - 1].axis('off')
    return fig
