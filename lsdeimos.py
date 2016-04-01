'''
LSDeimos: Deimos longslit data reduction

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

History:
1/25/2016 initial working (somewhat) version
'''

from __future__ import (absolute_import, division,
                        print_function, unicode_literals,
                        with_statement)
from builtins import (bytes, dict, int, list, object, range, str,
                      ascii, chr, hex, input, next, oct, open,
                      pow, round, super,
                      filter, map, zip)

import sys, os, glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit, minimize
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad
from scipy.signal import medfilt
from astropy.io import fits
from astropy.wcs import WCS
from astroscrappy import detect_cosmics

from lsd_utils import *

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


def normalize(fitsname, output=None, cr_remove=True, multiextension=True,
              masterflat="masterflat.fits", masterbias="masterbias.fits",
              cr_options=None):
    '''
    Flat field, bias subtraction, and cosmic ray removal with astroscrappy.
    
    Parameters:
    -----------
        fitsname: either string or 2d array, if string then name of fits file,
                  if 2d array, then the output of arrange_fits
        output: string, name of file to write out to (optional)
                if None, then just returns the array
        cr_remove: boolean, if true, the subtract cosmics
        multiextension: boolean, if true, then 
        masterflat: str, name of fits file with master flat (optional)
                    if None, then make a master flat from files in flatdir
        masterbias: str, name of fits file with master bias (optional)
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
        if cr_options is not None:
            for key, value in cr_options.items():
                kwargs[key] = value
        mask, cleaned = detect_cosmics(raw_data, **kwargs)
        normed_data = (cleaned - bias) / flat
    else:
        normed_data = (raw_data - bias) / flat


    if output is not None:
        fits.writeto(output, data=normed_data, header=header, clobber=True)
    
    return normed_data


def ap_trace(data, initial_guess=None, masterflat="masterflat.fits",
             nbins=20, nsigma=15, seeing=1., arcsec_per_pix=0.1185, frac_med=1.5, slit_width=1.):
    '''
    Traces the spatial apeture of a gaussian point source.

    Parameters:
    -----------
    data: 2d numpy array
    initial_guess: int, initial guess for the spatial position (here assumed as
                   x-axis) of the source
    masterflat: str, name of flat file to use
    nbins: int, number of spectral bins to fit trace
    nsigma: float, how far away from the peak should the trace look, in stdev
    seeing: float, in arcsec, used to get an initial sigma guess for the trace
    arcsec_per_pix: float, float, should get from header, default for DEIMOS
    frac_med: float, fraction of median image above which to mask (e.g., for sky lines)
    slit_width: float, arcsec, width of slit

    Returns:
    --------
    trace: function from spectral index (y) to spatial index (x)
    sigma: function from spectral index (y) to width in spatial index
    '''

    # mask maker mask maker make me a mask
    # True for bad pixels
    flat_mask = get_slitmask(masterflat=masterflat)
    # sky_mask = data > np.nanmedian(data) * frac_med
    # mask = np.logical_or(flat_mask, sky_mask)
    mask = flat_mask

    y_size, x_size = data.shape
    # try to do binning on smaller scale based on slit width
    nbins = int(y_size * arcsec_per_pix / slit_width / 10)
    spectral_bin_edges = np.linspace(0, y_size, nbins + 1).astype(int)
    bin_sizes = y_size / nbins
    spectral_bin_centers = np.linspace(bin_sizes * 0.5,
                                       bin_sizes * (nbins - 0.5) ,
                                       nbins).astype(int)
    box_size = round(seeing / arcsec_per_pix * nsigma)

    if initial_guess is None:
        # make a rough guess
        spatial = np.sum(data, axis=0)
        smooth = medfilt(spatial, kernel_size=5)
        # roughly captures how far we have to go into the chip for constant illumination
        frac = 0.05
        x = np.arange(smooth.size)
        spatial_mask = np.logical_or(x < x.size * frac, x > x.size * (1 - frac))
        smooth[spatial_mask] = np.nan
        initial_guess = np.nanargmax(smooth)
        
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
        popt, pcov = curve_fit(gaussian, x[mask], specbin[mask], p0=params)
        # plot for sanity check
        # space = np.linspace(0, x.max(), 1000)
        # plt.clf()
        # plt.plot(space, gaussian(space, *popt))
        # plt.plot(x[mask], specbin[mask], 'ko')
        # plt.show()

        # if err > 10**2, reject fit
        perr = np.sqrt(np.diag(pcov))
        # fit to gaussian and remember to add back cropped out spatial
        if perr[2] < 10**2:
            fit_centers[i] = popt[2] + x_low
        else:
            fit_centers[i] = np.nan
        fit_sigmas[i] = popt[3]

    mask = ~np.isnan(fit_centers)
    sigma_median = np.nanmedian(fit_sigmas)
    sigma_range = seeing / arcsec_per_pix
    mask = mask & (fit_sigmas < sigma_median + sigma_range)
    mask = mask & (fit_sigmas > sigma_median - sigma_range)

    # polynomial fit to fit centers and sigmas
    poly_center = np.polyfit(spectral_bin_centers[mask].astype(float),
                             fit_centers[mask].astype(float), deg=2,
                             w=1 / fit_sigmas[mask].astype(float)**2)
    poly_sigma = np.polyfit(spectral_bin_centers[mask].astype(float),
                            fit_sigmas[mask].astype(float), deg=2)

    trace = lambda y: np.polyval(poly_center, y)
    sigma = lambda y: np.polyval(poly_sigma, y)
    return trace, sigma

    # spline interpolation to get in between the binned spectral data
    # trace_spline = UnivariateSpline(spectral_bin_centers[mask], fit_centers[mask],
    #                                 k=3, s=1)
    # sigma_spline = UnivariateSpline(spectral_bin_centers[mask], fit_sigmas[mask],
    #                                 k=3, s=1)
    # return trace_spline, sigma_spline
    

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
    ap
    sky
    uncertainty
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
            pfit = np.polyfit(x_sky, sky_slice, skydeg)
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

    return aperture, sky, unc


def sky_wavesol(sky, header, deg=4):
    '''
    Calculate the wavelength solution from the sky calibration.

    Parameters
    ----------
    sky: float array, sky spectrum from ap_extract
    header: header from data file
    deg: int, degree of fit for poly mode

    Returns
    -------
    wfunc: function from trace center to wavelength, in Angstroms
    '''

    p0 = get_initial_wavesol(header)
    if len(p0) < deg + 1:
        # assuming that p0 refers to the lower degrees only
        p0 = np.append(p0, np.zeros(deg + 1 - len(p0)))
    grating_name = header['GRATENAM']
    
    sky_wave, sky_flux = get_sky_spectrum(grating_name)
    # function from wavelength in angstroms to sky flux
    f_ref = interp1d(sky_wave, sky_flux, bounds_error=False, fill_value=np.nan)

    # map pixel space to sky flux
    # I'm using lower to higher terms for fit, but polyval
    # uses higher to lower
    def poly_wave(y, p): return np.polyval(p[::-1], y)
    def fit_function(y, *p): return f_ref(poly_wave(y, p))
    return fit_function

    y = np.arange(sky.size)
    y0 = y.size / 2.
    
    # normalize to height
    sky_normed = sky * np.amax(sky_flux) / np.amax(sky)
    # w_init = poly_wave(y - y0, p0)
    # w_low = w_init[0]
    # w_high = w_init[-1]
    # ref_area = quad(f_ref, w_low, w_high)[0]
    # sky_area = np.sum(sky)
    # sky_normed = sky * ref_area / sky_area

    # subtract y pixels by zeros point at center
    popt, pcov = curve_fit(fit_function, y - y0, sky_normed, p0)
    def wfunc(y): return poly_wave(y - y0, popt)
    return wfunc


def wavelength_solution(trace_spl, sigma_spl, arc_name, lines="spec2d_lamp_NIST.dat",
                        mode='poly', deg=4, method='Nelder-Mead', linear_no_fit=False):
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
    method: str, optimization method to use
    linear_no_fit: bool, if true, then just return the inital guess from header

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

    if linear_no_fit:
        def wfunc(y): return w0 + w_per_pix * (y - y0)
        return wfunc
    
    list_lines, list_heights = good_lines(lines)
    
    # get pixel values of lines    
    ap_lines = ap_extract(arc_data, trace_spl, sigma_spl, sky_subtract=False)

    # list of arc lines
    arc_lines, arc_heights = get_lines(ap_lines)
    
    # array of zeroth degree and linear terms
    p0 = np.array([w0, w_per_pix])
    # append zero terms for higher order fits
    if len(p0) < deg + 1:
        # assuming that p0 refers to the lower degrees only
        p0 = np.append(p0, np.zeros(deg + 1 - len(p0)))
    
    if mode == 'poly':
        # I'm using lower to higher terms for fit, but polyval
        # uses higher to lower
        def fit_function(y, p): return np.polyval(p[::-1], y)
    else:
        raise NotImplementedError("Only mode='poly' is currently implemented.")

    # subtract y pixels by zeros point at center
    def min_function(p): return premetric(fit_function(arc_lines - y0, p), list_lines)    
    res = minimize(min_function, p0, method=method, options={'disp': True})

    def wfunc(y): return fit_function(y - y.shape[0]/2., res.x)
    return wfunc
    

def reduce_files(files, save=True, plot=True):

    for f in files:
        
        print("Cleaning", f)
        clean_name = f.split('.')[0] + '.clean.fits'
        if os.path.exists(clean_name):
            clean = fits.getdata(clean_name)
        else:
            clean = normalize(f, output=clean_name)
        continue
        print("Tracing", f)
        trace_spl, sigma_spl = ap_trace(clean)
        
        print("Extracting aperture and sky subtracting", f)
        ap, sky, unc = ap_extract(clean, trace_spl, sigma_spl)
        
        print("Fitting wavelength solution", f)
        wfunc = wavelength_solution(trace_spl, sigma_spl, "normed_NeArKrXe.fits")

        y = np.arange(ap.size)
        w = wfunc(y)
        if save:
            header = "Spectrum extracted from " + f + "\nAngstroms\tFlux"
            write_array = list(zip(w, ap - sky, unc))
            np.savetxt(f.split('.')[0] + ".dat", write_array, header=header)
        if plot:
            plt.clf()
            plt.plot(wfunc(y), ap - sky, 'k-')
            plt.xlabel("Wavelength (Angstroms)")
            plt.ylabel("Relatived flux (arbitrary units)")
            plt.show()
    
    
if __name__ == "__main__":
    reduce_files(sys.argv[1:])
        
