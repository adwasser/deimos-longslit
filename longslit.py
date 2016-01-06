'''
QaDDRL: Quick and Dirty DEIMOS Reduction of Longslit data.
Rough order of business:
    Put Deimos fits hdu files in their place
    Flat field and bias
    Wavelength solution
    Trace and continuum subtraction
12/18/15
    New plan: do reduction on individual chips (rather than trying to do the
    whole darn thing at once.
'''

import sys, os, re, glob
import numpy as np
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel
import scipy.signal
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
from astropy.io import fits
from astroscrappy import detect_cosmics

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


def remove_overscan(fitsname):
    '''
    Removes the overscan region using the values set in header.
    '''
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
        # remove overscan pixels
        data_arrays.append(hdulist[i].data[y_low: y_high, x_low: x_high])
    data_arrays = np.array(data_arrays)
    
    return data_arrays

def multiextension_to_array(data_arrays):
    '''
    Takes the eight DEIMOS ccd arrays and returns a single array oriented
    correctly.
    '''
    chips_per_layer = len(data_arrays) / 2
    # stacks on stacks on stacks
    bottom = np.concatenate(data_arrays[0: chips_per_layer], axis=1)
    tops = data_arrays[chips_per_layer:]
    # rotate by 180 degrees the top CCDs
    tops = [np.rot90(a, 2) for a in tops]
    top = np.concatenate(tops, axis=1)

    stack = np.concatenate((bottom, top), axis=0)
    return stack

def array_to_multiextension(data_array):
    '''
    Splits a single array back into the eight DEIMOS ccds.
    '''
    rows = 2
    columns = 4
    bottom, top = np.split(data_array, rows, axis=0)
    bottoms = np.split(bottom, columns, axis=1)
    tops = np.split(top, columns, axis=1)

    # unrotate by 180 deg
    tops = [np.rot90(a, 2) for a in tops]

    bottoms.append(tops)
    return bottoms
    
    
def open_deimos(fitsname, output=None, crop=False):
    '''
    Opens a DEIMOS fits mosaic and returns a list
    of 2d numpy arrays representing the 8 ccd chips.
    If crop=True, then remove the overscan regions first.
    If output is not None, write to a single fits file.
    '''
    # HDU list has f[0] being the primary HDU and f[1:9] being the image arrays
    hdulist = fits.open(fitsname)
    # need to crop if outputing as single fits!
    if crop or output is not None:
        data_arrays = remove_overscan(fitsname)
    else:
        data_arrays = np.array([hdulist[i].data for i in range(1, 9)])

    # I'm hard-coding in the layout of the DEIMOS chip now, with 8 CCDs in the
    # mosaic, 1-4 on the bottom and 5-8 on the top
    # The bottom ones have their origin in the lower left, the top ones have
    # their origin in the upper right.

    if output is not None:
        stack = multiextention_to_array(data_arrays)
        header = hdulist[0].header
        header.extend(deimos_cards(stack.shape))
        fits.writeto(output, stack, header=header, clobber=True)

    return data_arrays
    

def make_masterbias(output="masterbias.fits", biasdir="bias/", remove_cr=True):
    '''
    Creates a median bias file from fits files found in biasdir.
    '''
    bias_files = glob.glob(biasdir + "*.fits")
    bias_data = []
    # arbitrarily taking the first exposure as the header info
    hdulist = fits.open(bias_files[0])
    headers = [hdu.header for hdu in hdulist]
    for i, f in enumerate(bias_files):
        bias_data.append(np.array(open_deimos(f)))
    # axis 0 is different exposures, axis 1 is different CCDs
    # axis 2, 3 are y, x
    bias_data = np.array(bias_data)
    bias = np.median(bias_data, axis=0)
    if remove_cr:
        masks = []
        cr_cleaned = []
        for frame in bias:
            mask, cleaned = detect_cosmics(frame, verbose=True,
                                           cleantype='medmask',
                                           sigclip=0.2)
            masks.append(mask)
            cr_cleaned.append(cleaned)
        bias = cr_cleaned

    if output is not None:
        hdulist = []
        primary = fits.PrimaryHDU(header=headers[0])
        hdulist.append(primary)
        for i, frame in enumerate(bias):
            hdu = fits.ImageHDU(data=frame, header=headers[i + 1])
            hdulist.append(hdu)
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(output, clobber=True)
    return bias


def make_masterflat(output="masterflat.fits", flatdir="flats/", bias="masterbias.fits"):
    '''
    Creates a mean flat field from fits files found in flatdir.
    '''
    if isinstance(bias, str):
        bias = open_deimos(bias, crop=True)

    flat_files = glob.glob(flatdir + "*.fits")
    flat_data = []
    # arbitrarily taking the first exposure as the header info
    hdulist = fits.open(flat_files[0])
    headers = [hdu.header for hdu in hdulist]

    # fraction of median level to count as illuminated
    illum_threshold = 0.8

    for i, flat_file in enumerate(flat_files):
        # bias subtract the flats here
        subframes = open_deimos(flat_file, crop=True) - bias
        new_subframes = np.empty(subframes.shape)
        # norm out by median of illuminated pixels
        for j, subframe in enumerate(subframes):
            illuminated = np.where(subframe >= illum_threshold * np.median(subframe))
            normed = subframe / np.median(subframe[illuminated])
            new_subframes[j] = normed
        flat_data.append(new_subframes)
    # axis 0 is different exposures, axis 1 is different CCDs
    # axis 2, 3 are y, x
    flat_data = np.array(flat_data)
    flat = np.mean(flat_data, axis=0)

    # combine into one array for the lamp division
    flat = multiextension_to_array(flat)
    
    # divide out by flat field lamp spectrum
    spectral_size, spatial_size = flat.shape
    spectral_pixels = np.arange(spectral_size)
    spatial_pixels = np.arange(spatial_size)
    # response technique from PyDIS (https://github.com/jradavenport/pydis)
    flat_1d = np.log10(convolve(flat.sum(axis=1), Box1DKernel(5)))
    spline = UnivariateSpline(spectral_pixels, flat_1d, ext=0, k=2, s=0.001)
    flat_curve = 10.0 ** spline(spectral_pixels)
    # tile back up to shape of flat file
    flat_curve = np.tile(np.split(flat_curve, flat_curve.size, axis=0), (1, spatial_size))
    flat = flat / flat_curve

    # unpack back into multiextension format
    flat = array_to_multiextension(flat)
    
    if output is not None:
        hdulist = []
        primary = fits.PrimaryHDU(header=headers[0])
        hdulist.append(primary)
        for i, frame in enumerate(flat):
            hdu = fits.ImageHDU(data=frame, header=headers[i + 1])
            hdulist.append(hdu)
        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(output, clobber=True)

    return flat

    
def normalize(raw_data, singlefits=False, masterflat=None, masterbias=None,
              flatdir="./flats", biasdir="./bias", output=None):
    '''
    Flat field and bias subtraction.
    
    Parameters:
    -----------
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
    --------
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

    return normed_data


def ap_trace(data, spatial_pixel,
             nsteps=20, nsigma=15, seeing=1., arcsec_per_pix=0.1185):
    '''
    Traces the spatial apeture of a gaussian point source.

    Parameters:
    -----------
    data: 2d numpy array
    spatial_pixel: int, initial guess for the spatial position (here assumed as
                   x-axis) of the source
    nsteps: int, number of spectral bins to fit trace
    nsigma: float, how far away from the peak should the trace look, in stdev
    seeing: float, in arcsec, used to get an initial sigma guess for the trace
    arcsec_per_pix: float, float, should get from header, default for DEIMOS

    Returns:
    --------
    trace: 1d numpy array, spatial (x) pixel index for each spectral (y) point
    '''

    y_size, x_size = data.shape
    spectral_bins = np.linspace(0, y_size, nbins)

    for i in range(nbins - 1):
        # fit gaussian to spatial trace, flattening spectral axis
        pass
    

def wavelength_solution():
    pass

def wavelength_extract(wavesol, trace):
    pass

def ap_extract(data, trace):
    '''
    Extract the spectrum using a specified trace.
    '''
    pass
    

