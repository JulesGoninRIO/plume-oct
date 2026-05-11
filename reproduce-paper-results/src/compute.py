import numpy as np
import scipy
import scipy.fftpack
from numpy import pi, sin
from scipy.optimize import leastsq
from scipy.ndimage.interpolation import shift
from scipy.optimize import differential_evolution
import cv2

# From https://matthew-brett.github.io/teaching/mutual_information.html


def zero_padding(src, shape, pos):
    """
    Add zeros on the border of the images
    :param src: source image to pad
    :param shape: the shape to be padded to
    :param pos: the position of the padding ????
    :return: the image padded
    """
    y, x = (int(pos[0]), int(pos[1]))
    padded_image = np.zeros(shape)
    padded_image[y : src.shape[0] + y, x : src.shape[1] + x] = src
    return padded_image


def misalignment_model(al, dt1, dt2, misalignment, fitting_area):
    """
    Construct the misalignment model
    :param al: ??
    :param dt1: ??
    :param misalignment: ??
    :param fitting_area: ??
    :return: the misalignment model
    """
    N1, N2 = misalignment.shape
    V1, V2 = map(lambda x: 2 * x + 1, fitting_area)
    return (
        lambda n1, n2: al
        * sin((n1 + dt1) * V1 / N1 * pi)
        * sin((n2 + dt2) * V2 / N2 * pi)
        / (sin((n1 + dt1) * pi / N1) * sin((n2 + dt2) * pi / N2) * (N1 * N2))
    )


def misalignmentfunc(ref_image, cmp_image):
    """
    Evaluates the misalignment function for two images to find the offset
    :param ref_image: the reference image
    :param cmp_image: the image to register with
    :return: the value of the misalignment function
    """
    # Windowing to reduce boundary effects
    hanning_window_x = np.hanning(ref_image.shape[0])
    hanning_window_y = np.hanning(ref_image.shape[1])
    hanning_window_2d = (
        hanning_window_x.reshape(hanning_window_x.shape[0], 1) * hanning_window_y
    )
    ref_image, cmp_image = [ref_image, cmp_image] * hanning_window_2d

    # Fourier Transform and cross power
    F = scipy.fftpack.fft2(ref_image)
    G = scipy.fftpack.fft2(cmp_image)
    G_ = np.conj(G)

    # cross-phase spectrum
    R = F * G_ / np.abs(F * G_)
    # shit the zero-frequency domain component to the center
    R = scipy.fftpack.fftshift(R)
    # Spectral weighting technique to reduce aliasing and noise effects
    # M = [M1, M2]
    M = np.floor([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])
    # U = [U1, U2]
    U = M / 2.0
    low_pass_filter = np.ones([int(M[0]) + 1, int(M[1]) + 1])
    low_pass_filter = zero_padding(low_pass_filter, ref_image.shape, U)
    R = R * low_pass_filter
    R = scipy.fftpack.fftshift(R)

    # Reverse fourier
    misalignment = scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))
    return misalignment


def main_misalignment_reg(ref_image, cmp_image, fitting_shape=(8, 8), eps=0.001):
    """
    Runs the misalignment registration algorithm on two images
    :param ref_image: the reference image
    :param cmp_image: the image to register with
    :return dy: the offset in the y direction
    :return dx: the offset in the x direction
    :return match_height: the result of the last iteration
    """
    # calculate phase-only correlation
    misalignment = misalignmentfunc(ref_image, cmp_image)
    # get peak position peak
    max_pos = np.argmax(misalignment)
    peak = np.array(
        [max_pos / ref_image.shape[1], max_pos % ref_image.shape[1]]
    ).astype(int)

    # fitting using least-square method
    mc = np.array([fitting_shape[0] / 2.0, fitting_shape[1] / 2.0])
    fitting_area = misalignment[
        int(peak[0] - mc[0]) : int(peak[0] + mc[0] + 1),
        int(peak[1] - mc[1]) : int(peak[1] + mc[1] + 1),
    ]

    if fitting_area.shape != (fitting_shape[0] + 1, fitting_shape[1] + 1):
        return 0.0, 0.0, 0.0

    m = np.array([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])
    u = m / 2
    y, x = np.mgrid[-mc[0] : mc[0] + 1, -mc[1] : mc[1] + 1]
    y = np.ceil(y + peak[0] - m[0])
    x = np.ceil(x + peak[1] - m[1])

    # Error function
    def error_func(p):
        misalignment_model_values = misalignment_model(p[0], p[1], p[2], misalignment, u)(y, x)
        error = misalignment_model_values - fitting_area
        return np.ravel(error)

    # p0 to opitmized
    p0 = np.array([0.0, -(peak[0] - m[0]) - eps, -(peak[1] - m[1]) - eps])

    # LS optimization
    estimate = leastsq(error_func, p0, maxfev=100)
    match_height = estimate[0][0]

    # Extract refined displacement
    dx = -estimate[0][1]
    dy = -estimate[0][2]
    return dy, dx, match_height


# https://stackoverflow.com/questions/56357039/numpy-zero-padding-to-match-a-certain-shape


def error_func(p, misalignment, u, y, x, fitting_area):
    misalignment_model_values = misalignment_model(p[0], p[1], p[2], misalignment, u)(y, x)
    error = misalignment_model_values - fitting_area
    return np.ravel(error)


def a_scan_normalization(volume: np.ndarray) -> np.ndarray:
    """Compute the A-scan normalization of an OCT cube

    Args:
        volume (np.ndarray): OCT cube as 3D numpy array

    Returns:
        np.ndarray: column-normalized OCT cube
    """
    assert len(volume.shape) == 3, " Provided array has the wrong number of dimensions"
    return np.divide(
        (volume - volume.min(axis=1)[:, np.newaxis, :]),
        volume.ptp(axis=1)[:, np.newaxis, :],
        where=volume.ptp(axis=1)[:, np.newaxis, :] != 0,
    )
