import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_data(filename):
    wavelengths = []
    fluxes = []
    with open(filename, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                try:
                    wavelength, flux = line.split(',')
                    wavelengths.append(float(wavelength))
                    fluxes.append(float(flux))
                except ValueError:
                    continue
    return np.array(wavelengths), np.array(fluxes)

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def linear(x, a, b):
    return a * x + b

def combined_model(x, mu, sigma, amplitude, a, b):
    return gaussian(x, mu, sigma, amplitude) + linear(x, a, b)

def fit_spectrum(wavelengths, fluxes):
    # initial gusses
    mu_guess = wavelengths[np.argmax(fluxes)]
    sigma_guess = (wavelengths[-1] - wavelengths[0]) / 10
    amplitude_guess = np.max(fluxes)
    a_guess = (fluxes[-1] - fluxes[0]) / (wavelengths[-1] - wavelengths[0])
    b_guess = fluxes[0]

    p0 = [mu_guess, sigma_guess, amplitude_guess, a_guess, b_guess]

    # fit the data(curve fit)
    popt, pcov = curve_fit(combined_model, wavelengths, fluxes, p0=p0)
    return popt, pcov

def plot_spectrum(wavelengths, fluxes, popt):
    plt.figure(figsize=(12, 8))

    # plot all the data
    plt.subplot(3, 1, 1)
    plt.plot(wavelengths, fluxes, label='spectrum')
    plt.xlabel('wavelength (angstrom)')
    plt.ylabel('flux (ADU)')
    plt.title('full spectrum')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    background = linear(wavelengths, *popt[3:])
    plt.plot(wavelengths, fluxes, label='spectrum')
    plt.plot(wavelengths, background, label='background fit', linestyle='-')
    plt.xlabel('wavelength (angstrom)')
    plt.ylabel('flux (ADU)')
    plt.title('spectrum with background fit')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    combined_fit = combined_model(wavelengths, *popt)
    plt.plot(wavelengths, fluxes, label='spectrum')
    plt.plot(wavelengths, combined_fit, label='combined fit', linestyle='--',c="r")
    plt.plot(wavelengths, background, label='background fit', linestyle='--')
    plt.xlabel('wavelength (angstrom)')
    plt.ylabel('flux (ADU)')
    plt.title('spectrum with gaussian fit and background')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('C://My Files//Space Science//Space Detector Laboratory//Assignments//Assignment-1//spectrum_fitting_results.png')
    plt.show()

def main():
    # read spectrum data
    wavelengths, fluxes = read_data('C://My Files//Space Science//Space Detector Laboratory//Assignments//Assignment-1//spectrum.txt')

    # fit spectrum
    popt, _ = fit_spectrum(wavelengths, fluxes)

    # fit parameters
    mu, sigma, amplitude, a, b = popt
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    print(f"Fit Parameters:")
    print(f"Centroid (mu): {mu:.4f} Angstrom")
    print(f"Standard Deviation (sigma): {sigma:.4f} Angstrom")
    print(f"Amplitude (A): {amplitude:.4f} ADU")
    print(f"Background Slope (a): {a:.4f}")
    print(f"Background Intercept (b): {b:.4f}")
    print(f"Full Width at Half Maximum (FWHM): {fwhm:.4f} Angstrom")

    # plot results
    plot_spectrum(wavelengths, fluxes, popt)

if __name__ == "__main__":
    main()