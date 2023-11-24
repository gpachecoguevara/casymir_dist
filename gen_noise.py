import example
import numpy as np
import casymir.casymir
import casymir.processes
import matplotlib.pyplot as plt


# Sjoerd's functions for noise generation from 1D NNPS
def apply_nps(nps: np.array, freq: np.array, det_size: list, pixel_size: list):
    """
    Function that applies a given 1D nps to white noise
    :param nps: 1D nps
    :param freq: frequencies corresponding to nps
    :param det_size: detector size in pixels
    :param pixel_size: pixel size in mm
    :return: noise colored nps and with detector dimensions
    """
    Nx = det_size[0]
    Ny = det_size[1]

    # Obtained NPS, rescale size and magnitude
    freq2D = get_freq2D(Nx, Ny, pixel_size)

    # Get NPS value at each frequency
    NPS_fit = _closest(freq2D, nps, freq)

    # Normalize the field
    NPS_fit /= np.max(NPS_fit)
    NPS_fit[np.where(NPS_fit < 0.0)] = 0.0
    NPS_fit[np.where(NPS_fit > 1.0)] = 1.0

    # Shift zero frequency to the corners
    NPS_fit = np.roll(NPS_fit, int(Ny / 2), axis=1)
    NPS_fit = np.roll(NPS_fit, int(Nx / 2), axis=0)

    # Scale white noise
    white_noise = np.random.randn(Nx, Ny)
    white_noise -= np.mean(white_noise)
    white_noise = (1.0 / np.sqrt(np.var(white_noise))) * white_noise

    # Apply NPS
    color_noise_FFT = np.fft.fft2(white_noise) * np.sqrt(NPS_fit)
    color_noise = np.real(np.fft.ifft2(color_noise_FFT))

    return color_noise


def _closest(freq2D: np.array, nps: np.array, freq: np.array):
    """
    Function that assigns linearly interpolates nps to the desired frequencies in the 2D frequency field
    :param freq2D: 2D frequency field
    :param nps: NPS points
    :param freq: frequency of the NPS points
    :return: 2D NPS field corresponding to the given NPS and frequency field
    """
    NPS_fit = np.zeros_like(freq2D, np.float32)
    for x in range(0, freq2D.shape[0]):
        for y in range(0, freq2D.shape[1]):
            f = freq2D[x, y]
            if f >= np.max(freq):
                NPS_fit[x, y] = nps[-1]
            else:
                diff = np.abs(freq - f)
                ind = np.argpartition(np.abs(diff), 1)[0:2]
                val = np.partition(np.abs(diff), 1)[0:2]
                NPS_fit[x, y] = nps[ind[0]] * (val[1] / (val[0] + val[1])) + nps[ind[1]] * (val[0] / (val[0] + val[1]))

    return NPS_fit


def get_freq2D(Naxis_w: int, Naxis_d: int, pixel_size):
    """
    This function returns the 2D frequency map in 1/mm (NOT 1/pixel)
    :param pixel_size: pixel size
    :param Naxis_w: width of detector
    :param Naxis_d: depth of detector
    :return: matrix with 2D frequency values
    """
    axis_w = np.transpose(np.tile(np.linspace(-Naxis_w / 2 + 1, Naxis_w / 2, Naxis_w), (Naxis_d, 1))) / \
             (Naxis_w * pixel_size[0])
    axis_d = np.tile(np.linspace(-Naxis_d / 2 + 1, Naxis_d / 2, Naxis_d), (Naxis_w, 1)) / \
             (Naxis_d * pixel_size[1])
    freq2D = np.sqrt(axis_w ** 2 + axis_d ** 2)
    return freq2D


# CASYMIR implementation to get 1D NNPS (and MTF)
print_fit_params = "N"              # Change to Y if you need a polynomial fit for the NNPS
spectrum = "Example DBT Spectrum"   # Name of the spectrum
kV = 28
mAs = 40
system = "example_dbt.yaml"         # Name of the file containing all DBT system parameters (don't change).

sys = casymir.casymir.System(system)

if sys.detector["type"] == "direct":
    results = example.run_model_dc(system, spectrum, kV, mAs, fit=print_fit_params)

else:
    results = example.run_model_ic(system, spectrum, kV, mAs, fit=print_fit_params)

# To save results to xlsx file (we only need the numpy array to generate the images)
# example.save_to_excel(results, output_path)

# Plot NNPS
plt.figure()
plt.plot(results[:, 0], results[:, 2])
plt.xlabel("Frequency [1/mm]", {'size': 15})
plt.ylabel("Magnitude", {'size': 15})
plt.title("NNPS", {'size': 15})
plt.grid()
plt.show()

# SLOW. Real image size is (3164, 2364). Set it to these values to test it works
noise = apply_nps(nps=results[:, 2], freq=results[:, 0], det_size=[316, 236],
                  pixel_size=[sys.detector["px_size"], sys.detector["px_size"]])
# Show noise image
plt.figure()
plt.imshow(noise, cmap="gray")
plt.show()
