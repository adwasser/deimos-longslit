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

import sys, os, glob
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from astropy.convolution import convolve, Box1DKernel, Box2DKernel
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import SmoothBivariateSpline
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
        return np.nan


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


def multiextension_to_array(fitsname=None, data_arrays=None, headers=None):
    '''
    Takes the eight DEIMOS ccd arrays and returns a single array oriented
    correctly.  Supply the list of headers to insure the correct orientation.
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
    wcslist = [WCS(header) for header in headers]

    # get the mosaic image size from any of the headers 
    det_x0, det_x1, det_y0, det_y1 = get_indices(headers[0]['DETSIZE'])
    mosaic = np.empty(shape=(det_y1, det_x1))
    mosaic[:] = np.nan

    for i, data in enumerate(arrays):
        '''
        I'm assuming the wcs transformation is perfectly linear, so I can just
        specify the start and end pixels.
        '''
        # select the data which isn't part of the overscan region
        x0, x1, y0, y1 = get_indices(headers[i]['DATASEC'])
        # get the direction of pixel increments
        dx = float(headers[i]['CD1_1'])
        dy = float(headers[i]['CD2_2'])
        
        # subtract by 1 to convert from FITS format 1-based indexing to 0-based
        indices = np.index_exp[y0 - 1: y1, x0 - 1: x1]
        start = [x0 - 1, y0 - 1]
        end = [x1, y1]
        old_coords = np.array([start, end])
        
        # subtract 0.5 to convert from DEIMOS plane wcs definition to indices
        new_coords = wcslist[i].all_pix2world(old_coords, 0) - 0.5
        new_start, new_end = map(list, new_coords)
        # need to handle going backwards in the slice
        new_end[0] = None if new_end[0] < 0 else new_end[0]
        new_end[1] = None if new_end[1] < 0 else new_end[1]        
        
        # remember to switch from x, y in FITS to y, x in numpy
        new_indices = np.index_exp[new_start[1]: new_end[1]: dy,
                                   new_start[0]: new_end[0]: dx]
        mosaic[new_indices] = data[indices]

    return mosaic
    

def make_masterbias(output="masterbias.fits", biasdir="bias/", remove_cr=True):
    '''
    Creates a median bias file from fits files found in biasdir.
    '''
    bias_files = glob.glob(biasdir + "*.fits")
    # arbitrarily taking the first exposure as the header info
    hdulist = fits.open(bias_files[0])
    primary_header = hdulist[0].header
    headers = [hdu.header for hdu in hdulist[1:9]]

    bias_data = []
    # axis 0 is different exposures, axis 1 is different CCDs
    # axis 2, 3 are y, x
    for i, f in enumerate(bias_files):
        bias_data.append(np.array([hdu.data for hdu in fits.open(f)[1:9]]))
    bias_data = np.array(bias_data)

    # if we have more bias frames, we should cr subtract before medianing biases
    bias = np.nanmedian(bias_data, axis=0)
    if remove_cr:
        masks = []
        cr_cleaned = []
        for frame in bias:
            mask, cleaned = detect_cosmics(frame, verbose=True,
                                           cleantype='medmask',
                                           sigclip=0.2)
            masks.append(mask)
            cr_cleaned.append(cleaned)
        bias = np.array(cr_cleaned)
        
    bias = multiextension_to_array(data_arrays=bias, headers=headers)
    if output is not None:
        header = primary_header
        header.extend(deimos_cards(bias.shape))
        hdu = fits.PrimaryHDU(data=bias,
                              header=header)
        hdu.writeto(output, clobber=True)
    return bias


def make_masterflat(output="masterflat.fits", flatdir="flats/", bias="masterbias.fits"):
    '''
    Creates a mean flat field from fits files found in flatdir.
    '''
    if isinstance(bias, str):
        bias = fits.getdata(bias)

    flat_files = glob.glob(flatdir + "*.fits")
    flat_data = np.array([multiextension_to_array(f) for f in flat_files])

    # arbitrarily taking the first exposure as the header info
    hdulist = fits.open(flat_files[0])
    headers = [hdu.header for hdu in hdulist]

    # divide by median after subtracting bias
    for i, data in enumerate(flat_data):
        flat_data[i] -= bias
        flat_data[i] /= np.nanmedian(flat_data[i])
    # median across all frames
    flat = np.nanmedian(flat_data, axis=0)

    # normalize to lamp response
    # response technique from PyDIS (https://github.com/jradavenport/pydis)
    spectral_size, spatial_size = flat.shape
    spectral_pixels = np.arange(spectral_size)
    spatial_pixels = np.arange(spatial_size)
    flat_1d = np.log10(convolve(flat.sum(axis=1), Box1DKernel(5)))
    spline = UnivariateSpline(spectral_pixels, flat_1d, ext=0, k=2, s=0.001)
    flat_curve = 10.0 ** spline(spectral_pixels)
    # tile back up to shape of flat file
    flat_curve = np.tile(np.split(flat_curve, flat_curve.size, axis=0), (1, spatial_size))
    flat /= flat_curve

    if output is not None:
        header = headers[0]
        header.extend(deimos_cards(flat.shape))
        hdu = fits.PrimaryHDU(data=flat,
                              header=header)
        hdu.writeto(output, clobber=True)
    return flat

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

def normalize(fitsname, output=None, cr_remove=True, multiextension=True,
              masterflat="masterflat.fits", masterbias="masterbias.fits"):
    '''
    Flat field and bias subtraction.
    
    Parameters:
    -----------
        :fitsname: either string or 2d array, if string then name of fits file,
                   if 2d array, then the output of arrange_fits
        :output: string, name of file to write out to (optional)
                 if None, then just returns the array
        :cr_remove: boolean, if true, the subtract cosmics
        :multiextension: boolean, if true, then 
        :masterflat: str, name of fits file with master flat (optional)
                     if None, then make a master flat from files in flatdir
        :masterbias: str, name of fits file with master bias (optional)
                     if None, then make a master bias from files in biasdir

    Returns:
    --------
        :normalized_data: 2d numpy array, rows are spectral, columns are spatial
    '''
    if masterbias is None:
        bias = make_masterbias()
    else:
        bias = fits.getdata(masterbias)
    if masterflat is None:
        flat = make_masterflat()
    else:
        flat = fits.getdata(masterflat)

    if multiextension:
        raw_data = multiextension_to_array(fitsname)
        header = fits.open(fitsname)[0].header
        header.extend(deimos_cards(raw_data.shape))
    else:
        raw_data = fits.getdata(fitsname)
        header = fits.getheader(fitsname)
        
    if cr_remove:
        flat_mask = get_slitmask(flat)
        mask, cleaned = detect_cosmics(raw_data, verbose=True,
                                       inmask=flat_mask,
                                       cleantype='medmask',
                                       sigclip=0.5, sigfrac=0.1, niter=4)
        normed_data = (cleaned - bias) / flat
    else:
        normed_data = (raw_data - bias) / flat


    if output is not None:
        fits.writeto(output, data=normed_data, header=header, clobber=True)
    
    return normed_data


def data_combine(files, masterbias='masterbias.fits', masterflat='masterflat.fits'):
    '''
    Sum a list of images.
    '''
    
    data = np.array([fits.getdata(f) for f in files])
    # median out each frame
    frame_medians = np.nanmedian(data, axis=(1,2))[:, np.newaxis, np.newaxis]
    data = data / frame_medians
    # get median of frames
    median = np.nanmedian(data, axis=0)

    return data.sum(axis=0)

def _gaussian(x, a, b, x0, sigma):
    # Gaussian function
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

    
def ap_trace(data, initial_guess,
             nbins=20, nsigma=15, seeing=1., arcsec_per_pix=0.1185, frac_med=1.5):
    '''
    Traces the spatial apeture of a gaussian point source.

    Parameters:
    -----------
    data: 2d numpy array
    initial_guess: int, initial guess for the spatial position (here assumed as
                   x-axis) of the source
    nbins: int, number of spectral bins to fit trace
    nsigma: float, how far away from the peak should the trace look, in stdev
    seeing: float, in arcsec, used to get an initial sigma guess for the trace
    arcsec_per_pix: float, float, should get from header, default for DEIMOS
    frac_med: float, fraction of median image above which to mask (e.g., for sky lines)

    Returns:
    --------
    trace: 1d numpy array, spatial (x) pixel index for each spectral (y) point
    '''

    # mask maker mask maker make me a mask
    # True for bad pixels
    flat_mask = get_slitmask(masterflat='masterflat.fits')
    sky_mask = data > np.nanmedian(data) * frac_med
    mask = np.logical_or(flat_mask, sky_mask)
    
    y_size, x_size = data.shape
    spectral_bin_edges = np.linspace(0, y_size, nbins + 1).astype(int)
    bin_sizes = y_size / nbins
    spectral_bin_centers = np.linspace(bin_sizes * 0.5,
                                       bin_sizes * (nbins - 0.5) ,
                                       nbins).astype(int)
    box_size = round(seeing / arcsec_per_pix * nsigma)
    x_low = initial_guess - box_size
    x_high = initial_guess + box_size

    crop = data[:, x_low: x_high]
    mask = mask[:, x_low: x_high]
    crop[mask] = np.nan

    specbin_arrays = np.array_split(crop, spectral_bin_edges[1:-1])
    specbins = np.empty(shape=(nbins, x_high - x_low))
    for i, array in enumerate(specbin_arrays):
        specbins[i] = np.nanmean(array, axis=0)

    fit_centers = np.empty(specbins.shape[0])
    x = np.arange(specbins.shape[1])
    sigma = seeing / arcsec_per_pix
    sky_limit = round(3 * sigma)
    for i, specbin in enumerate(specbins):
        mask = ~np.isnan(specbin)
        a = np.nanmax(specbin[mask])
        x0 = np.nanargmax(specbin)
        sky = np.concatenate((specbin[:x0 - sky_limit], specbin[x0 + sky_limit:]))
        b = np.nanmedian(sky)
        params = [a, b, x0, sigma]
        popt, pcov = curve_fit(_gaussian, x[mask], specbin[mask], p0=params)
        # plot for sanity check
        # space = np.linspace(0, x.max(), 1000)
        # plt.plot(space, _gaussian(space, *popt))
        # plt.plot(x[mask], specbin[mask], 'ko')
        # plt.show()

        # if err > 10**2, reject fit
        perr = np.sqrt(np.diag(pcov))
        # fit to gaussian and remember to add back cropped out spatial indices
        fit_centers[i] = popt[2] + x_low if perr[2] < 10**2 else np.nan

    mask = ~np.isnan(fit_centers)
    # spline interpolation to get in between the binned spectral data
    ap_spline = UnivariateSpline(spectral_bin_centers[mask], fit_centers[mask],
                                 ext=0, k=3, s=1)
    return fit_centers, spectral_bin_centers, ap_spline
    
    
    
def wavelength_solution():
    pass

def wavelength_extract(wavesol, trace):
    pass

def ap_extract(data, trace):
    '''
    Extract the spectrum using a specified trace.
    '''
    pass
    

