'''
Helper functions for longslit reduction.
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals,
                        with_statement)
from builtins import (bytes, dict, int, list, object, range, str,
                      ascii, chr, hex, input, next, oct, open,
                      pow, round, super,
                      filter, map, zip)

import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from astropy.convolution import convolve, MexicanHat1DKernel
from scipy.signal import medfilt
from astropy.io import fits
from astropy.wcs import WCS
from astroscrappy import detect_cosmics

def deimos_cards(shape, bunit='Counts'):
    '''
    Record fits header information.
    '''
    header = {}
    header['NAXIS'] = 2
    # switch between numpy row-column convention and fits column-row convention
    header['NAXIS1'] = shape[1]
    header['NAXIS2'] = shape[0]        
    header['BUNIT'] = bunit
    header['WCSNAME'] = 'array'
    header['CRPIX1'] = 1
    header['CRPIX2'] = 1        
    header['CRVAL1'] = 0
    header['CRVAL2'] = 0
    header['CDELT1'] = 1
    header['CDELT2'] = 1
    header['CTYPE1'] = 'Spatial_Pixel'
    header['CTYPE2'] = 'Dispersion_Pixel'
    return header.items()


def grating_to_disp(name):
    '''
    Returns the dispersion in angstroms per pixel for a given DEIMOS grating
    '''
    if name == '600ZD':
        return 0.65
    elif name == '830G':
        return 0.47
    elif name == '900ZD':
        return 0.44
    elif name == '1200G':
        return 0.33
    else:
        print(name, "is not a known grating!")
        sys.exit()


def get_indices(datasec):
    '''
    Takes a string from the fits header in the form '[x1:x2, y1:y2]' and returns
    a list with [x1, x2, y1, y2].
    '''
    xs, ys = [s.split(':') for s in datasec.strip('[]').split(',')]
    xs.extend(ys)
    return map(float, xs)
    

def remove_overscan(fitsname=None, data_arrays=None, headers=None):
    '''
    Removes the overscan region using the values set in header.
    '''
    rows = 2
    columns = 4
    nimages = rows * columns

    if fitsname is not None:
        hdulist = fits.open(fitsname)
        primaryHDU = hdulist[0]
        headers = [hdu.header for hdu in hdulist[1: nimages + 1]]
        arrays = [hdu.data for hdu in hdulist[1: nimages + 1]]
    elif data_arrays is not None and headers is not None:
        arrays = data_arrays        
    else:
        raise KeyError('Need either fitsname or both arrays and headers!')

    new_arrays = []
    # there are eight images
    for i in range(1, 9):
        y_size, x_size = arrays[i].shape
        # get the usable pixels in the image, defined in the DATASEC keyword
        datasec = headers[i]['DATASEC']
        x_low, x_high, y_low, y_high = get_indices(datasec)
        # need to adjust the low values to account for zero-based indexing
        x_low, y_low = x_low - 1., y_low - 1.
        # remove overscan pixels
        new_arrays.append(arrays[i][y_low: y_high, x_low: x_high])
    new_arrays = np.array(new_arrays)
    
    return new_arrays


def get_slitmask(masterflat='masterflat.fits'):
    # mask where not spatially illuminated due to bad slit coverage
    # also mask any spatial columns with saturated pixels
    if isinstance(masterflat, str):
        flat = fits.getdata(masterflat)
    else:
        # assume given the array directly
        flat = masterflat
    spatial = np.sum(flat, axis=0)
    # smooth with a kernel size designed to remove slit gaps and bad columns
    smoothed = medfilt(spatial, kernel_size=151)
    bad = np.logical_or(spatial < 0.9 * smoothed, spatial > 1.1 * smoothed)
    flat_mask = np.tile(bad, (flat.shape[0], 1))    
    return flat_mask


def gaussian(x, a, b, x0, sigma):
    # Gaussian function
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def closest(item, array):
    idx = np.argmin(np.abs(array - item))
    return array[idx]


def premetric(array1, array2):
    '''
    This is a janky way of measuring how "close" two lists of floats are.
    Not a metric, we fail symmetry and triangle inequality for simple cases
    of different sized arrays.
    '''
    s = 0
    for x in array1:
        s += np.abs(x - closest(x, array2))
    return s
    

def good_lines(master_line_list="spec2d_lamp_NIST.dat"):
    '''
    Returns a list of line wavelengths to check against.
    I have no idea what height refers to, but seems to work...
    '''
    specdat = np.loadtxt(master_line_list, dtype=str)
    wave, height = specdat[:, 0:2].astype(float).T
    qual = specdat[:, 2]
    good_line_indices = np.logical_and(qual == "GOOD", height > 500)
    return wave[good_line_indices], height[good_line_indices]
    

def get_lines(ap_lines):
    '''
    ap_lines is a 1d array with lots of peaks.  return the indices of those peaks!
    '''
    smooth = convolve(ap_lines, MexicanHat1DKernel(2))
    y = np.arange(ap_lines.shape[0])
    w = smooth > 0.1 * smooth.max()
    
    pixels = y[w]
    s = smooth[w]
    
    # go through pixels with spec above threshold, and look for successive pixels
    centers = []
    heights = []
    spreads = []
    run = [pixels[0]]
    for i, pixel in enumerate(pixels[1:]):
        # i gives (index in y) - 1
        # check if this pixel is one more than previous
        
        if run[-1] + 1 == pixel:
            run.append(pixel)
        else:
            # then this pixel is in the next run
            # process the run and put the pixel in the new run
            # print(run)
            run_array = np.array(run)
            center = np.mean(run_array)
            centers.append(center)
            heights.append(ap_lines[round(center)])
            spreads.append(len(run))
            # reset
            run = [pixel]
        if i == len(y) - 2:
            # then this is the last pixel
            # process the run and end loop
            # print(run)
            run_array = np.array(run)
            center = np.mean(run_array)
            centers.append(center)
            heights.append(smooth[round(center)])
            spreads.append(len(run))
    # now process all the lines found and fit with a gaussian to find a better center
    fit_centers = []
    fit_heights = []
    for i in range(len(centers)):
        a = heights[i]
        b = 10**-2 * centers[i]
        x0 = centers[i]
        sigma = spreads[i] / 2.
        p0 = [a, b, x0, sigma]
        indices = slice(round(x0 - 3 * sigma), round(x0 + 3 * sigma))
        popt, pcov = curve_fit(gaussian, y[indices], ap_lines[indices], p0)
        fit_centers.append(popt[2])
        fit_heights.append(popt[0])
        # sanity plots
        # plt.clf()
        # plt.plot(y[indices], gaussian(y[indices], *popt), 'r-')
        # plt.plot(y[indices], ap_lines[indices], 'ko')
        # plt.show()
    return np.array(fit_centers), np.array(fit_heights)
            
    
