'''
QaDDRL: Quick and Dirty DEIMOS Reduction of Longslit data.
Rough order of business:
    Put Deimos fits hdu files in their place
    Flat field and bias
    Wavelength solution
    Trace and continuum subtraction
'''

import sys, os, re
import numpy as np
from astropy.io import fits

def deimos_cards(shape, bunit='Counts'):
    '''
    Record fits header information.
    '''
    header = {}
    header['NAXIS'] = 2
    header['NAXIS1'] = shape[0]
    header['NAXIS2'] = shape[1]        
    header['BUNIT'] = bunit
    header['CRPIX1'] = 1
    header['CRPIX2'] = 1        
    header['CRVAL1'] = 0.5
    header['CRVAL2'] = 0.5
    header['CDELT1'] = 1
    header['CDELT2'] = 1
    header['CTYPE1'] = 'Spatial Pixel'
    header['CTYPE2'] = 'Dispersion Pixel'
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
        return np.nan
        
def undeimos(fitsname, output=None):
    '''
    Takes the standard DEIMOS packaging of fits header data units (hdu's) and
    rearranges them into one 2d array with x and y corresponding to physical
    dimesions (roughly spatial and spectral, respectively).
    
    Parameters:
    -------------
        :fitsname: string, name of fits file written to by DEIMOS
        :output: string, name of file to write out to (optional)
                 if None, then just returns the array
    Returns:
    ------------
        :raw_data: 2d numpy array, rows are spectral, columns are spatial
    '''
    # f is an HDU list, with f[0] being the primary HDU and f[1:9] being the
    # image arrays
    hdulist = fits.open(fitsname)
    data_arrays = []
    # Everybody stand back.  I know regular expressions.
    pattern  = re.compile('\[([0-9]+):([0-9]+),([0-9]+):([0-9]+)\]')
    # there are eight images
    for i in range(1, 9):
        y_size, x_size = hdulist[i].shape
        # get the usable pixels in the image, defined in the DATASEC keyword
        datasec = hdulist[i].header['DATASEC']
        strings = pattern.match(datasec).group(1, 2, 3, 4)
        x_low, x_high, y_low, y_high = map(float, strings)
        # need to adjust the low values to account for zero-based indexing
        x_low, y_low = x_low - 1., y_low - 1.
        data_arrays.append(hdulist[i].data[y_low: y_high, x_low: x_high])

    # I'm hard-coding in the layout of the DEIMOS chip now, with 8 CCDs in the
    # mosaic, 1-4 on the bottom and 5-8 on the top
    # The bottom ones have their origin in the lower left, the top ones have
    # their origin in the upper right.

    chips_per_layer = len(data_arrays) / 2
    # rotate by 180 degrees the top CCDs
    for i in range(chips_per_layer, len(data_arrays)):
        data_arrays[i] = np.rot90(data_arrays[i], 2)
        
    # stacks on stacks on stacks
    bottom = np.concatenate(data_arrays[0: chips_per_layer], axis=1)
    top = np.concatenate(data_arrays[chips_per_layer:], axis=1)
    stack = np.concatenate((bottom, top), axis=0)

    if output is not None:
        header = hdulist[0].header
        header.extend(deimos_cards(stack.shape))
        fits.writeto(output, stack, header=header, clobber=True)
        
    return stack

def make_masterbias(output="masterbias.fits", biasdir="./bias"):
    '''
    Creates a median bias file from fits files found in biasdir.
    '''
    bias_files = os.listdir(biasdir)
    bias_data = []
    for i, f in enumerate(bias_files):
        bias_data.append(undeimos(biasdir + "/" + f))
    bias = np.median(np.array(bias_data), axis=0)
    hdu = fits.PrimaryHDU()
    hdu.data = bias
    hdu.header.extend(deimos_cards(bias.shape))
    hdu.writeto(output, clobber=True)
    return bias
    
def make_masterflat(output="masterflat.fits", flatdir="./flats", bias="masterbias.fits"):
    '''
    Creates a mean flat field from fits files found in flatdir.
    '''
    if isinstance(bias, str):
        bias = fits.getdata(bias)
        
    flat_files = os.listdir(flatdir)
    flat_data = []
    for i, f in enumerate(flat_files):
        flat_data.append(undeimos(flatdir + "/" + f))
    # subtract master bias from mean flat
    flat = np.mean(np.array(flat_data), axis=0) - bias
    hdu = fits.PrimaryHDU()
    hdu.data = flat
    hdu.header.extend(deimos_cards(flat.shape))
    hdu.writeto(output, clobber=True)
    return flat

    
def normalize(raw_data, singlefits=True, masterflat=None, masterbias=None,
              flatdir="./flats", biasdir="./bias", output=None):
    '''
    Flat field and bias subtraction.
    
    Parameters:
    -------------
        :data: either string or 2d array, if string then name of fits file,
               if 2d array, then the output of arrange_fits
        :singlefits: boolean, if false, then the data filename refers to the
                     original DEIMOS arrangement
        :masterflat: str, name of fits file with master flat (optional)
                     if None, then make a master flat from files in flatdir
        :masterbias: str, name of fits file with master bias (optional)
                     if None, then make a master bias from files in biasdir
        :flatdir: string, name of directory with flat field fits files (optional)
        :biasdir: string, name of directory with bias fits files (optional)
        :output: string, name of file to write out to (optional)
                 if None, then just returns the array
    Returns:
    ------------
        :normalized_data: 2d numpy array, rows are spectral, columns are spatial
    '''
    if isinstance(raw_data, str):
        if singlefits:
            data = fits.getdata(raw_data)
        else:
            data = undeimos(raw_data)

    if masterbias is None:
        bias = make_masterbias(biasdir=biasdir)
    else:
        bias = fits.getdata(masterbias)
        
    if masterflat is None:
        flat = make_masterflat(flatdir=flatdir, bias=bias)
    else:
        flat = fits.getdata(masterflat)
        
    # normalize flat field
    flat = flat / np.mean(flat)

    normed_data = (data - bias) / flat

    if output is not None:
        hdu = fits.PrimaryHDU()
        if isinstance(raw_data, str):
            header = fits.getheader(raw_data)
            if not singlefits:
                header.extend(deimos_cards(normed_data.shape, 'Normalized Counts'))
        else:
            header = hdu.header
            header.extend(deimos_cards(normed_data.shape, 'Normalized Counts'))

        flat_history = masterflat if masterflat is not None else flatdir
        bias_history = masterbias if masterbias is not None else biasdir
        header['HISTORY'] = 'Flat frame: ' + flat_history        
        header['HISTORY'] = 'Bias frame: ' + bias_history
        hdu.data = normed_data
        hdu.header = header
        hdu.writeto(output, clobber=True)       

    return flat
    return normed_data


def wavelength_solution():
    pass

