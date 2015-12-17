'''
QaDDRL: Quick and Dirty DEIMOS Reduction of Longslit data.
Rough order of business:
    Put Deimos fits hdu files in their place
    Flat field and bias
    Wavelength solution
    Trace and continuum subtraction
'''

import sys, re
import numpy as np
from astropy.io import fits

def arrange_fits(fitsname, save_to=None):
    '''
    Takes the standard DEIMOS packaging of fits header data units (hdu's) and
    rearranges them into one 2d array with x and y corresponding to physical
    dimesions (roughly spatial and spectral, respectively).
    
    Parameters:
    -------------
        :fitsname: string, name of fits file written to by DEIMOS
        :save_to: string, name of file to write out to (optional)
                  if None, then just returns the array
    Returns:
    ------------
        :raw_data: 2d numpy array, rows are spectral, columns are spatial
    '''
    # f is an HDU list, with f[0] being the primary HDU and f[1:9] being the
    # image arrays
    f = fits.open(fitsname)
    data_arrays = []
    # Everybody stand back.  I know regular expressions.
    pattern  = re.compile('\[([0-9]+):([0-9]+),([0-9]+):([0-9]+)\]')
    for i in range(1: len(f)):
        y_size, x_size = f[i].shape
        # get the usable pixels in the image, defined in the DATASEC keyword
        datasec = f[i].header['DATASEC']
        strings = pattern.match(datasec).group(1, 2, 3, 4)
        x_low, x_high, y_low, y_high = map(float, strings)
        # need to adjust the low values to account for zero-based indexing
        x_low, y_low = x_low - 1., y_low - 1.
        data_array.append(f[i].data[x_low: x_high, y_low: y_high])
    pass

def normalize(data, flatdir="./flats", biasdir="./bias", save_to=None):
    '''
    Flat field and bias subtraction.
    
    Parameters:
    -------------
        :data: either string or 2d array, if string then name of fits file,
               if 2d array, then the output of arrange_fits
        :flatdir: string, name of directory with flat field fits files (optional)
        :biasdir: string, name of directory with bias fits files (optional)
        :save_to: string, name of file to write out to (optional)
                  if None, then just returns the array
    Returns:
    ------------
        :normalized_data: 2d numpy array, rows are spectral, columns are spatial
    '''
    pass

def wavelength_solution():
    pass

