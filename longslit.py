from __future__ import (absolute_import, division,
                        print_function, unicode_literals,
                        with_statement)
from builtins import (
         bytes, dict, int, list, object, range, str,
         ascii, chr, hex, input, next, oct, open,
         pow, round, super,
         filter, map, zip)

'''
DRoLs: DEIMOS Reduction of Longslit data

What this is: see name above.
What this is not: bulletproof spectroscopic reduction.  This was made solely to
get reasonably looking wavelength solutions for the 600ZD grating DEIMOS longslit
data we obtained that seemed to break to "official" spec2d pipeline.  This does
\it{not} do flux calibration.  Use at your own peril.

Dependencies: the usual astropy suite (numpy, scipy, matplotlib, astropy)
along with astroscrappy for cosmic ray subtraction

Development notes:
The DEIMOS chip layout is shown in the engineering drawings found here:
http://www.ucolick.org/~sla/fits/mosaic/d0307j.pdf

After getting a single array with the (as currently named)
multiextension_to_array function, the dispersion axis will be in the y-axis
(columns) and the spatial axis will be in x (rows).  Wavelength increases with
y indices.

Much of the inspiration for this comes from the PyDIS package, found here:
https://github.com/jradavenport/pydis
'''

import sys, os, glob
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from astropy.convolution import convolve, Box1DKernel, MexicanHat1DKernel
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
    spline = UnivariateSpline(spectral_pixels, flat_1d, k=2, s=0.001)
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
              masterflat="masterflat.fits", masterbias="masterbias.fits",
              cr_options=None):
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
        :cr_options: dict, keys should be keywords of detect_cosmics from
                     astroscrappy, overrides defaults below
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
        kwargs = {'verbose': True, 'inmask': flat_mask, 'cleantype': 'medmask',
                  'sigclip': 0.5, 'sigfrac': 0.1, 'niter': 4}
        for key, value in cr_options.items():
            kwargs[key] = value
        mask, cleaned = detect_cosmics(raw_data, **kwargs)
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
    # sky_mask = data > np.nanmedian(data) * frac_med
    # mask = np.logical_or(flat_mask, sky_mask)
    mask = flat_mask
    
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
    fit_sigmas = np.empty(specbins.shape[0])

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
        # plt.clf()
        # plt.plot(space, _gaussian(space, *popt))
        # plt.plot(x[mask], specbin[mask], 'ko')
        # plt.show()

        # if err > 10**2, reject fit
        perr = np.sqrt(np.diag(pcov))
        # fit to gaussian and remember to add back cropped out spatial indices
        if perr[2] < 10**2:
            fit_centers[i] = popt[2] + x_low
        else:
            fit_centers[i] = np.nan
        fit_sigmas[i] = popt[3]

    mask = ~np.isnan(fit_centers)
    # spline interpolation to get in between the binned spectral data
    trace_spline = UnivariateSpline(spectral_bin_centers[mask], fit_centers[mask],
                                    ext=0, k=3, s=1)
    sigma_spline = UnivariateSpline(spectral_bin_centers[mask], fit_sigmas[mask],
                                    ext=0, k=3, s=1)
    return trace_spline, sigma_spline
    

def ap_extract(data, trace_spl, sigma_spl,
               apwidth=2, skysep=1, skywidth=2, skydeg=0, sky_subtract=True):
    '''
    Extract the spectrum using a specified trace.
    Data is the 2d array, trace_spl, sigma_spl are the splines from ap_trace.

    Parameters
    -----------
    :apwidth: in factors of sigma, corresponds to aperature radius
    :skysep: in factors of sigma, corresponds to the separation between sky and aperature
    :skywidth: in factors of sigma, corresponds to the width of sky windows on either side
    :skydeg: degree of polynomial fit for the sky spatial profile at each wavelength

    Returns
    -------
    source
    sky
    ap_err
    '''

    y_size, x_size = data.shape
    y_indices, x_indices = np.indices(data.shape)
    y_bins = np.arange(y_size)
    x_bins = np.arange(x_size)
    x_centers = trace_spl(y_bins)
    x_sigmas = sigma_spl(y_bins)

    # specify aperature as a function of spectral position
    ap_lows = x_centers - apwidth * x_sigmas
    ap_highs = x_centers + apwidth * x_sigmas

    right_sky_lows = x_centers - (apwidth + skysep + skywidth) * x_sigmas
    right_sky_highs = x_centers - (apwidth + skysep) * x_sigmas
    left_sky_lows = x_centers + (apwidth + skysep) * x_sigmas
    left_sky_highs = x_centers + (apwidth + skysep + skywidth) * x_sigmas

    ap_pixels = np.logical_and(ap_lows < x_indices, x_indices < ap_highs)
    right_sky_pixels = np.logical_and(right_sky_lows < x_indices, x_indices < right_sky_highs)
    left_sky_pixels = np.logical_and(left_sky_lows < x_indices, x_indices < left_sky_highs)    
    sky_pixels = np.logical_or(right_sky_pixels, left_sky_pixels)

    aperture = np.empty(y_size)
    sky = np.empty(y_size)
    unc = np.empty(y_size)

    if not sky_subtract:
        # just return aperture sum
        for i in range(y_size):
            aperture[i] = np.nansum(data[i, ap_pixels[i]])
        return aperture
    
    for i in range(y_size):
        data_slice = data[i]
        ap_slice = data_slice[ap_pixels[i]]
        aperture[i] = np.nansum(ap_slice)
        sky_slice = data_slice[sky_pixels[i]]
        x_sky = x_bins[sky_pixels[i]]
        x_ap = x_bins[ap_pixels[i]]
        if skydeg > 0:
            pfit = np.polyfit(x_sky, sky, skydeg)
            sky[i] = np.nansum(np.polyval(pfit, x_ap))
        elif skydeg == 0:
            sky[i] = np.nanmean(sky_slice) * (apwidth * x_sigmas[i] * 2 + 1)

        # for now...
        coaddN = 1
        # uncertainty, as done by PyDIS
        sigB = np.std(sky_slice) # stddev in the background data
        N_B = len(x_sky) # number of bkgd pixels
        N_A = apwidth * x_sigmas[i] * 2. + 1 # number of aperture pixels
        # based on aperture phot err description by F. Masci, Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        unc[i] = np.sqrt(np.sum((aperture[i] - sky[i]) / coaddN) +
                         (N_A + N_A**2. / N_B) * (sigB**2.))

    source = aperture - sky

    return aperture, sky, unc


def _closest(item, array):
    idx = np.argmin(np.abs(array - item))
    return array(idx)


def _premetric(array1, array2):
    '''
    This is a janky way of measuring how "close" two lists of floats are.
    Not a metric, we fail symmetry and triangle inequality for simple cases.
    '''
    s = 0
    for x in array1:
        s += np.abs(x - _closest(x, array2))
    return s
    

def _good_lines(master_line_list="spec2d_lamp_NIST.dat"):
    '''
    Returns a list of line wavelengths to check against.
    I have no idea what height refers to, but seems to work...
    '''
    specdat = np.loadtxt(master_line_list, dtype=str)
    wave, height = specdat[:, 0:2].astype(float).T
    qual = specdat[:, 2]
    good_line_indices = np.logical_and(qual == "GOOD", height > 500)
    return wave[good_line_indices]
    

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
    for i in range(len(centers)):
        a = heights[i]
        b = 10**-2 * centers[i]
        x0 = centers[i]
        sigma = spreads[i] / 2.
        p0 = [a, b, x0, sigma]
        indices = slice(round(x0 - 3 * sigma), round(x0 + 3 * sigma))
        popt, pcov = curve_fit(_gaussian, y[indices], ap_lines[indices], p0)
        fit_centers.append(popt[2])
        # sanity plots
        # plt.clf()
        # plt.plot(y[indices], _gaussian(y[indices], *popt), 'r-')
        # plt.plot(y[indices], ap_lines[indices], 'ko')
        # plt.show()
    return fit_centers
            
    

def wavelength_solution(trace_spl, sigma_spl, arc_name, lines="spec2d_lamp_NIST.dat",
                        mode='poly', deg=2):
    '''
    Calculate the wavelength solution, given an arcline calibration frame.
    I'm assuming that the initial guess is very good (i.e., that we can identify
    lines based on which are the closest) and that there are more lines in
    arc_lines than list_lines.

    Parameters
    ----------
    trace_spl: trace spline (i.e., from ap_trace)
    sigma_spl: sigma spline (from ap_trace)
    arc_name: str, name of fits file with arc frame
    lines: str, name of file with lines (in Angstroms) as the first column
    mode: str, poly for polynomial fit
    deg: int, degree of fit for poly mode

    Returns
    -------
    wfunc: function from trace center to wavelength, in Angstroms
    '''

    header = fits.getheader(arc_name)
    arc_data = fits.getdata(arc_name)
    grating_name = header['GRATENAM']
    grating_position = header['GRATEPOS']
    # this should be the nominal wavelength at the center of the detector
    # if I'm reading the info at the page below correctly
    # http://www2.keck.hawaii.edu/inst/deimos/grating-info.html
    w0 = header['G' + str(grating_position).strip() + 'TLTWAV']
    w_per_pix = grating_to_disp(grating_name)
    y0 = arc_data.shape[0] / 2

    list_lines = _good_lines(lines)
    
    # get pixel values of lines    
    ap_lines = ap_extract(arc_data, trace_spl, sigma_spl, sky_subtract=False)

    # need to implement
    arc_lines = get_lines(ap_lines)
    
    # array of zeroth degree and linear terms
    p0 = np.array([w0, w_per_pix])
    # append zero terms for higher order fits
    if len(p0) < deg + 1:
        # assuming that p0 refers to the lower degrees only
        p0 = np.append(p0, np.zeros(deg + 1 - len(p0)))
    
    if mode == 'poly':
        # I'm using lower to higher terms for fit, but polyval
        # uses higher to lower
        def fit_function(x, p): return np.polyval(p[::-1], x)
    else:
        raise NotImplementedError("Only mode='poly' is currently implemented.")

    def min_function(p): return _premetric(fit_function(arc_lines, p), list_lines)    
    popt, pcov = minimize(min_function, p0)

    return popt, pcov
    

