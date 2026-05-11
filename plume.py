DCM_PATH = r"V:\Studies\_FINISHED\Uveitis\data\Heyex\Heyex_OCT_cube\301261\15"
DCM_PATHf = "your path to oct dicom here"

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from scipy.stats import norm
import matplotlib.image as mpimg
import scipy.fftpack
from numpy import pi, sin
from scipy.optimize import leastsq
from loguru import logger


class PLUME:
    """
    Minimal PLUME: compute misalignment-derived scores + SNR for ONE OL OCT_CUBE dataset.
    No IO side effects (no CSV/PKL, no plots, no saving).
    """

    def __init__(self):
        return

    # ---------- public API ----------

    def score_dataset(self, dataset_folder, oct_cube_info):
        """
        Args:
            dataset_folder: path to the dataset folder (the one containing oct/volume)
            info: parsed dataset info.json (your browse already loads this)
            oct_cube_info: parsed oct/volume/info.json (your browse already loads this)

        Returns:
            dict with plume metrics, or None if not applicable (not OCT_CUBE)
        """
        z_factor, _, x_factor, y_factor = oct_cube_info["spacing"]
        if z_factor == 0:
            logger.info(f'PLUME, z_factor==0: {dataset_folder}')
            return None

        volume = load_array_from_folder(dataset_folder)
        if volume is None or volume.size <= 1 or volume.shape == (1, 1):
            logger.warning(f'PLUME, invalide oct volume: {dataset_folder}')
            return None

        n_b_scans = int(volume.shape[0])

        misalignment, misalignment_weighted = self.misalignment_volume(volume, n_b_scans)
        snr_b_scans = self.compute_SNR(volume)
        score_snr = float(np.mean(snr_b_scans)) if snr_b_scans is not None else np.nan

        tot_displacement = self.quality_score(misalignment, z_factor, x_factor, y_factor)
        spatially_weighted_displacement = self.quality_score(misalignment_weighted, z_factor, x_factor, y_factor)
        
        plume_result = {
            'n_b_scans': n_b_scans,
            'tot_displacement': float(tot_displacement),
            'weighted_displacement': float(spatially_weighted_displacement),
            'snr': float(score_snr),
        }
        return plume_result

    # ---------- internals ----------

    def generate_gaussian(self, n_b_scans: int, factor: int):
        scale = n_b_scans / 10
        center = n_b_scans / 2
        x = np.linspace(0, n_b_scans - 2, n_b_scans - 1)
        return factor * (scale) * norm.pdf(x, loc=center, scale=scale)

    def misalignment_volume(self, volume: np.ndarray, n_b_scans: int, factor: int = 10):
        volume_N = a_scan_normalization(volume)

        dx, dy = [], []
        for i in range(volume.shape[0] - 1):
            dx_, dy_, _ = main_misalignment_reg(volume_N[i], volume_N[i + 1])
            dx.append(dx_)
            dy.append(dy_)

        misalignment = np.abs(dx) + np.abs(dy)
        gaussian = self.generate_gaussian(n_b_scans, factor)
        misalignment_weighted = misalignment * gaussian
        return misalignment, misalignment_weighted

    def quality_score(self, misalignment: np.ndarray, z_factor: float, x_factor: float, y_factor: float):
        heights = float(np.sum(misalignment))
        factors = (float(x_factor) * float(y_factor)) / float(z_factor)
        return heights * factors

    def compute_SNR(self, volume: np.ndarray, cutoff_percentage=0.2, min_n_lines=2, db=True):
        assert len(volume.shape) == 3, "Provided array has the wrong number of dimensions"

        is_identical = np.std(volume, axis=2) == 0
        res = np.zeros(len(volume), dtype=float) + 1.0

        for i in range(volume.shape[0]):
            bscan = volume[i, ~is_identical[i, :], :]
            if bscan.shape[0] < min_n_lines:
                continue
            sorted_lines_i = np.argsort(np.mean(bscan, axis=1))
            k = max(int(cutoff_percentage * bscan.shape[0]), min_n_lines)
            background = bscan[sorted_lines_i[:k]]
            roi = bscan[sorted_lines_i[-k:]]
            res[i] = float(np.mean(roi) / np.std(background)) if np.std(background) != 0 else 1.0

        if db:
            res = 20 * np.log10(res)
        return res

def load_array_from_folder(folder_path: str) -> np.ndarray:
    """Load grayscale OCT volume as numpy array from a given string path.

    Returns:
        np.ndarray: OCT cube as numpy array. If a loading error occurs, returns np.empty((1, 1)).
    """
    # Safer than folder_path += "/" (works on Windows too)
    if not os.path.isdir(folder_path):
        return np.empty((1, 1))

    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and f.lower().endswith((".jpg", ".png"))
    ]

    # ✅ Handle empty folder / no matching files
    if not files:
        return np.empty((1, 1))

    # Longest file name
    max_len = len(max(files, key=len))
    # Recompute files number as three digits
    files_extended = ["0" * (max_len - len(f)) + f for f in files]
    # Sort files by number
    files = [x for _, x in sorted(zip(files_extended, files))]

    first_path = os.path.join(folder_path, files[0])
    try:
        image_shape = mpimg.imread(first_path).shape
    except BaseException:
        return np.empty((1, 1))

    volume = np.empty((len(files), image_shape[0], image_shape[1]))
    for index, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            try:
                volume[index] = mpimg.imread(file_path)
            except BaseException:
                return np.empty((1, 1))
        else:
            return np.empty((1, 1))

    return volume.astype("uint8")

# -------------------- Compute Plume --------------------

# --- caches keyed by shape ---
_HANNING_CACHE = {}
_LOWPASS_CACHE = {}


def zero_padding(src, shape, pos):
    y, x = (int(pos[0]), int(pos[1]))
    padded_image = np.zeros(shape, dtype=src.dtype)  # avoid implicit float64
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


def _get_hanning_2d(shape):
    w = _HANNING_CACHE.get(shape)
    if w is None:
        hx = np.hanning(shape[0])
        hy = np.hanning(shape[1])
        w = hx[:, None] * hy[None, :]
        _HANNING_CACHE[shape] = w
    return w

def _get_lowpass_filter(shape):
    f = _LOWPASS_CACHE.get(shape)
    if f is None:
        # exactly the same construction you do now
        M = np.floor([shape[0] / 2.0, shape[1] / 2.0])
        U = M / 2.0
        base = np.ones([int(M[0]) + 1, int(M[1]) + 1], dtype=np.float64)
        f = zero_padding(base, shape, U)
        _LOWPASS_CACHE[shape] = f
    return f

def misalignmentfunc(ref_image, cmp_image):
    # Windowing (cached)
    w = _get_hanning_2d(ref_image.shape)
    ref_w = ref_image * w
    cmp_w = cmp_image * w

    # Fourier Transform and cross power
    F = scipy.fftpack.fft2(ref_w)
    G = scipy.fftpack.fft2(cmp_w)
    FG = F * np.conj(G)

    # cross-phase spectrum (same formula, fewer temporaries)
    denom = np.abs(FG)
    R = FG / denom

    # shift, apply low-pass, shift back (filter cached)
    R = scipy.fftpack.fftshift(R)
    R *= _get_lowpass_filter(ref_image.shape)
    R = scipy.fftpack.fftshift(R)

    # Reverse fourier
    misalignment = scipy.fftpack.fftshift(np.real(scipy.fftpack.ifft2(R)))
    return misalignment


def _residual_inline(p, N1, N2, V1, V2, y, x, fitting_area):
    al, dt1, dt2 = p

    # exactly the same expression as misalignment_model(...)(y, x)
    num = al * sin((y + dt1) * V1 / N1 * pi) * sin((x + dt2) * V2 / N2 * pi)
    den = sin((y + dt1) * pi / N1) * sin((x + dt2) * pi / N2) * (N1 * N2)

    return np.ravel((num / den) - fitting_area)

def main_misalignment_reg(ref_image, cmp_image, fitting_shape=(8, 8), eps=0.001):
    misalignment = misalignmentfunc(ref_image, cmp_image)

    # Peak (slightly faster + clearer)
    peak0, peak1 = np.unravel_index(np.argmax(misalignment), misalignment.shape)
    peak = np.array([peak0, peak1], dtype=int)

    mc = np.array([fitting_shape[0] / 2.0, fitting_shape[1] / 2.0])
    fitting_area = misalignment[
        int(peak[0] - mc[0]) : int(peak[0] + mc[0] + 1),
        int(peak[1] - mc[1]) : int(peak[1] + mc[1] + 1),
    ]

    if fitting_area.shape != (fitting_shape[0] + 1, fitting_shape[1] + 1):
        return 0.0, 0.0, 0.0

    m = np.array([ref_image.shape[0] / 2.0, ref_image.shape[1] / 2.0])
    u = m / 2
    N1, N2 = misalignment.shape
    V1, V2 = (2 * u[0] + 1), (2 * u[1] + 1)

    # same grid as before
    y, x = np.mgrid[-mc[0] : mc[0] + 1, -mc[1] : mc[1] + 1]
    y = np.ceil(y + peak[0] - m[0])
    x = np.ceil(x + peak[1] - m[1])

    p0 = np.array([0.0, -(peak[0] - m[0]) - eps, -(peak[1] - m[1]) - eps])

    # residual with reduced overhead
    def error_func_fast(p):
        return _residual_inline(p, N1, N2, V1, V2, y, x, fitting_area)

    estimate = leastsq(error_func_fast, p0, maxfev=100)
    match_height = estimate[0][0]
    dx = -estimate[0][1]
    dy = -estimate[0][2]
    return dy, dx, match_height

def error_func(p, misalignment, u, y, x, fitting_area):
    misalignment_model_values = misalignment_model(p[0], p[1], p[2], misalignment, u)(y, x)
    error = misalignment_model_values - fitting_area
    return np.ravel(error)


def a_scan_normalization(volume: np.ndarray) -> np.ndarray:
    assert len(volume.shape) == 3, "Provided array has the wrong number of dimensions"

    vmin = volume.min(axis=1, keepdims=True)
    vptp = np.ptp(volume, axis=1, keepdims=True)

    out = np.empty_like(volume, dtype=np.result_type(volume, np.float32))
    np.subtract(volume, vmin, out=out)
    np.divide(out, vptp, out=out, where=vptp != 0)
    return out

def load_dcm_volume(dcm_path: str) -> np.ndarray:
    """
    Load OCT volume from a DICOM file or from a folder containing DICOM files.
    Returns shape: (n_b_scans, height, width)
    """

    if os.path.isdir(dcm_path):
        files = [
            os.path.join(dcm_path, f)
            for f in os.listdir(dcm_path)
            if os.path.isfile(os.path.join(dcm_path, f))
        ]

        if not files:
            raise FileNotFoundError(f"No files found in folder: {dcm_path}")

        # Try reading all DICOM files
        slices = []
        for file in sorted(files):
            try:
                ds = pydicom.dcmread(file)
                arr = ds.pixel_array
                if arr.ndim == 2:
                    slices.append(arr)
                elif arr.ndim == 3:
                    slices.extend(arr)
            except Exception:
                continue

        if not slices:
            raise ValueError(f"No readable DICOM pixel data found in folder: {dcm_path}")

        volume = np.stack(slices, axis=0)

    else:
        ds = pydicom.dcmread(dcm_path)
        volume = ds.pixel_array

        if volume.ndim == 2:
            volume = volume[np.newaxis, :, :]
        elif volume.ndim != 3:
            raise ValueError(f"Unsupported DICOM pixel shape: {volume.shape}")

    return normalize_to_uint8(volume)


def normalize_to_uint8(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)

    vmin = np.min(volume)
    vmax = np.max(volume)

    if vmax == vmin:
        return np.zeros_like(volume, dtype=np.uint8)

    volume = (volume - vmin) / (vmax - vmin)
    volume = volume * 255

    return volume.astype(np.uint8)

def plot_fundus(volume: np.ndarray, results: dict):
    """
    Plot square-looking fundus reconstruction with PLUME metrics.
    """

    assert len(volume.shape) == 3, "Provided array has the wrong number of dimensions"

    fundus = np.mean(volume, axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Force image to occupy square space
    ax.imshow(
        fundus,
        cmap="gray",
        aspect="auto",   # <- important
    )

    ax.axis("off")

    title = (
        "Fundus reconstruction from OCT\n\n"
        f"B-scans: {results['n_b_scans']}    "
        f"Total: {results['tot_displacement']:.2f}    "
        f"Weighted: {results['weighted_displacement']:.2f}    "
        f"SNR: {results['snr']:.2f} dB"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # Fill almost entire square figure
    ax.set_position([0.05, 0.08, 0.90, 0.75])

    return fig


def main():
    volume = load_dcm_volume(DCM_PATH)

    oct_cube_info = {
        "spacing": [1.0, 1.0, 1.0, 1.0]  # z, ?, x, y
    }

    plume = PLUME()

    n_b_scans = int(volume.shape[0])
    misalignment, misalignment_weighted = plume.misalignment_volume(volume, n_b_scans)
    snr_b_scans = plume.compute_SNR(volume)

    z_factor = oct_cube_info["spacing"][0]
    x_factor = oct_cube_info["spacing"][2]
    y_factor = oct_cube_info["spacing"][3]

    results = {
        "n_b_scans": n_b_scans,
        "tot_displacement": float(
            plume.quality_score(misalignment, z_factor, x_factor, y_factor)
        ),
        "weighted_displacement": float(
            plume.quality_score(misalignment_weighted, z_factor, x_factor, y_factor)
        ),
        "snr": float(np.mean(snr_b_scans)),
    }

    print(results)

    fig = plot_fundus(volume, results)
    plt.show()


if __name__ == "__main__":
    main()