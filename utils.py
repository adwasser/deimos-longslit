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

LSD_DIR = '/Users/asher/work/software/lsdeimos/'

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
        raise ValueError(name + " is not a known grating!")
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
            

def get_initial_wavesol(header):
    '''
    Retrieves the first two terms (constant and linear) from the fits header.
    '''
    grating_name = header['GRATENAM']
    # this should be the nominal wavelength at the center of the detector
    # if I'm reading the info at the page below correctly
    # http://www2.keck.hawaii.edu/inst/deimos/grating-info.html
    grating_position = header['GRATEPOS']
    w0 = header['G' + str(grating_position).strip() + 'TLTWAV']
    w_per_pix = grating_to_disp(grating_name)
    p0 = np.array([w0, w_per_pix])
    return p0
    

def get_sky_spectrum(grating='1200G'):
    if grating == '1200G':
        sky_file = LSD_DIR + 'sky_spectra/mkea_sky_newmoon_DEIMOS_1200_2011oct.fits.gz'
    elif grating == '600ZD':
        sky_file = LSD_DIR + 'sky_spectra/mkea_sky_newmoon_DEIMOS_600_2011oct.fits.gz'
    else:
        raise ValueError('Grating name must be either "1200G" or "600ZD".')
    f = fits.open(sky_file)
    wavelengths = f[0].data
    fluxes = f[1].data
    return wavelengths, fluxes

def list_lines():
    '''
    String containing arc lines from spec2d_lamp_NIST.dat
    '''
    arc_data = (
    '''
    # Generated by arclines_nist.pro - D. P. Finkbeiner 2002-Jul-23
    # NOTE THAT THESE ARE _AIR_ WAVELENGTHS
     3001.7175   150 BLEND Ne
     3017.3602   120   BAD Ne
     3027.0656   300 BLEND Ne
     3028.9120   300 BLEND Ne
     3034.5115   120 BLEND Ne
     3047.6059   120 BLEND Ne
     3057.4389   300 BLEND Ne
     3088.2207   120 BLEND Ne
     3092.9484   120 BLEND Ne
     3118.2025   120 BLEND Ne
     3126.2452   120 BLEND Ne
     3128.9195   300 BLEND Xe
     3141.3761   300 BLEND Ne
     3194.6225   120 BLEND Ne
     3198.6314   500 BLEND Ne
     3209.4004   120 BLEND Ne
     3213.7782   120 BLEND Ne
     3214.3710   150 BLEND Ne
     3218.2370   150 BLEND Ne
     3224.8612   120 BLEND Ne
     3229.6159   120 BLEND Ne
     3230.1127   200 BLEND Ne
     3230.4636   120 BLEND Ne
     3232.0642   120 BLEND Ne
     3232.4161   150 BLEND Ne
     3245.7374   300 BLEND Kr
     3264.8522   150 BLEND Kr
     3297.7671   150 BLEND Ne
     3309.7818   150 BLEND Ne
     3319.7651   300 BLEND Ne
     3323.7760  1000 BLEND Ne
     3325.7954   200 BLEND Kr
     3327.1940   150 BLEND Ne
     3334.8779   200 BLEND Ne
     3344.4383   150 BLEND Ne
     3345.4960   300 BLEND Ne
     3345.8719   150 BLEND Ne
     3355.0584   200 BLEND Ne
     3357.8616   120 BLEND Ne
     3360.6388   200 BLEND Ne
     3362.2054   120 BLEND Ne
     3366.7641   200   BAD Xe
     3367.2580   120 BLEND Ne
     3369.8493   490 BLEND Ne
     3369.9491  1500 BLEND Ne
     3378.2580   500 BLEND Ne
     3388.4602   150 BLEND Ne
     3388.9780   120 BLEND Ne
     3392.8390   300 BLEND Ne
     3406.9871   120 BLEND Ne
     3416.9543   120 BLEND Ne
     3417.7291   120 BLEND Ne
     3417.9440  3200 BLEND Ne
     3418.0465   350 BLEND Ne
     3423.9528   150 BLEND Ne
     3428.7271   120  GOOD Ne
     3447.7427  6600 BLEND Ne
     3450.8047   970 BLEND Ne
     3454.2346  4100 BLEND Ne
     3460.5639  1700 BLEND Ne
     3464.3782  2400 BLEND Ne
     3466.6182  2900 BLEND Ne
     3472.6104  7500 BLEND Ne
     3479.5581   150 BLEND Ne
     3480.7698   200 BLEND Ne
     3481.9725   200 BLEND Ne
     3498.1029  1800 BLEND Ne
     3501.2552  2700 BLEND Ne
     3507.4554   200 BLEND Kr
     3510.7598   540 BLEND Ne
     3515.2293  2600 BLEND Ne
     3520.5102 17000 BLEND Ne
     3542.8827   120 BLEND Ne
     3557.8406   120 BLEND Ne
     3568.5377   250  GOOD Ne
     3574.6490   200 BLEND Ne
     3593.5634  4100 BLEND Ne
     3593.6768  2100 BLEND Ne
     3600.2062  1200  GOOD Ne
     3609.2160   140 BLEND Ne
     3624.1134   600 BLEND Xe
     3631.9252   200 BLEND Kr
     3633.7011  1600 BLEND Ne
     3643.9639   150 BLEND Ne
     3653.9642   250  GOOD Kr
     3664.1094   200 BLEND Ne
     3669.0410   150 BLEND Kr
     3682.2783   640 BLEND Ne
     3685.7714  1100 BLEND Ne
     3694.2471   200 BLEND Ne
     3701.2605  1100  GOOD Ne
     3709.6558   150 BLEND Ne
     3713.1149   250 BLEND Ne
     3718.0575   300   BAD Kr
     3718.6304   200 BLEND Kr
     3721.3846   150 BLEND Kr
     3727.1400   250 BLEND Ne
     3741.6730   200 BLEND Kr
     3744.8302   150 BLEND Kr
     3754.2502   230 BLEND Ne
     3765.3045   150 BLEND Ar
     3766.2923   800 BLEND Ne
     3777.1683  1000 BLEND Ne
     3778.1230   500 BLEND Kr
     3781.0502   300 BLEND Xe
     3783.1286   500 BLEND Kr
     3829.7848   120  GOOD Ne
     3875.4742   150   BAD Kr
     3877.8336   200 BLEND Xe
     3888.6781   500   BAD He
     3888.6814   500   BAD He
     3906.2088   150 BLEND Kr
     3920.1130   200 BLEND Kr
     3922.5813   500 BLEND Xe
     3950.6236   300 BLEND Xe
     4042.9242   150 BLEND Ar
     4050.0962   200 BLEND Xe
     4057.0673   300 BLEND Kr
     4057.4942   200   BAD Xe
     4065.1581   300 BLEND Kr
     4072.0342   200 BLEND Ar
     4088.3667   500  GOOD Kr
     4098.7588   250 BLEND Kr
     4103.9414   150  GOOD Ar
     4131.7528   300 BLEND Ar
     4145.1511   250 BLEND Kr
     4158.0665   200   BAD Xe
     4158.6194   400 BLEND Ar
     4180.1305  1000   BAD Xe
     4193.1769   500   BAD Xe
     4198.3455   200 BLEND Ar
     4200.7028   400 BLEND Ar
     4208.5127   300   BAD Xe
     4213.7512   300   BAD Xe
     4219.7696   150 BLEND Ne
     4223.0287   300   BAD Xe
     4238.2745   400   BAD Xe
     4245.4125   500   BAD Xe
     4250.6081   150 BLEND Kr
     4250.6731   120 BLEND Ne
     4259.3897   200  GOOD Ar
     4272.1961   150 BLEND Ar
     4273.9967  1000 BLEND Kr
     4277.5557   550 BLEND Ar
     4292.9504   600 BLEND Kr
     4296.4285   500   BAD Xe
     4300.5174   200 BLEND Kr
     4310.5346   500   BAD Xe
     4317.8326   500   BAD Kr
     4318.5774   400 BLEND Kr
     4319.6061  1000 BLEND Kr
     4323.0112   150   BAD Kr
     4330.5491  1000   BAD Xe
     4331.2259   200 BLEND Ar
     4348.0903   800 BLEND Ar
     4355.5032  3000 BLEND Kr
     4362.6673   500 BLEND Kr
     4369.2285   200   BAD Xe
     4369.7183   200 BLEND Kr
     4369.8903   120 BLEND Ne
     4370.7790   200 BLEND Ar
     4376.1476   800 BLEND Kr
     4379.5786   150 BLEND Ne
     4379.6926   150 BLEND Ar
     4386.5637   300   BAD Kr
     4392.0212   200 BLEND Ne
     4393.2219   500   BAD Xe
     4395.7912   500   BAD Xe
     4398.0205   150 BLEND Ne
     4399.9910   200 BLEND Kr
     4401.0117   200 BLEND Ar
     4406.9081   200   BAD Xe
     4409.3274   150 BLEND Ne
     4416.0956   150   BAD Xe
     4422.5462   300 BLEND Ne
     4424.8353   300 BLEND Ne
     4425.4256   150 BLEND Ne
     4426.0268   400 BLEND Ar
     4430.2137   150 BLEND Ar
     4430.9215   150   BAD Ne
     4431.1344   150   BAD Ne
     4431.7103   500 BLEND Kr
     4436.8379   600 BLEND Kr
     4448.1568   500   BAD Xe
     4453.9422   600 BLEND Kr
     4457.0743   120 BLEND Ne
     4462.2129  1000   BAD Xe
     4463.7145   800 BLEND Kr
     4471.4954   200   BAD He
     4471.4991   200   BAD He
     4471.4994   200   BAD He
     4475.0394   800 BLEND Kr
     4480.8878   500   BAD Xe
     4481.8355   200 BLEND Ar
     4483.2208   150 BLEND Ne
     4488.1183   300 BLEND Ne
     4489.9053   400   BAD Kr
     4502.3778   600 BLEND Kr
     4523.1661   400   BAD Kr
     4536.3252   150 BLEND Ne
     4537.7017   300 BLEND Ne
     4537.7758  1000 BLEND Ne
     4538.3264   300 BLEND Ne
     4545.0761   400 BLEND Ar
     4556.6369   200   BAD Kr
     4575.0848   300 BLEND Ne
     4577.2333   800 BLEND Kr
     4579.3737   400 BLEND Ar
     4582.0641   150 BLEND Ne
     4582.4746   150 BLEND Ne
     4583.0017   300 BLEND Kr
     4589.9218   400 BLEND Ar
     4592.8270   150   BAD Kr
     4609.5904   550 BLEND Ar
     4609.9350   150 BLEND Ne
     4615.3158   500 BLEND Kr
     4619.1897  1000 BLEND Kr
     4628.3326   150 BLEND Ne
     4633.9087   800 BLEND Kr
     4645.4407   300 BLEND Ne
     4656.4160   300 BLEND Ne
     4657.9241   400 BLEND Ar
     4658.8988  2000 BLEND Kr
     4661.1282   150 BLEND Ne
     4678.2412   300 BLEND Ne
     4679.1580   150 BLEND Ne
     4680.4289   500 BLEND Kr
     4694.3830   200 BLEND Kr
     4702.5526   150 BLEND Ne
     4704.4177  1500 BLEND Ne
     4708.8820  1200 BLEND Ne
     4710.0895  1000 BLEND Ne
     4712.0867  1500 BLEND Ne
     4715.3679  1500 BLEND Ne
     4726.8901   550 BLEND Ar
     4734.1741   600 BLEND Xe
     4735.9276   300 BLEND Ar
     4739.0248  3000 BLEND Kr
     4749.5955   300 BLEND Ne
     4752.7537   500 BLEND Ne
     4758.7474   150 BLEND Ne
     4762.4573   300 BLEND Kr
     4764.8867   800 BLEND Ar
     4765.7664  1000 BLEND Kr
     4780.3594   300 BLEND Ne
     4788.9473  1000 BLEND Ne
     4790.2392   500 BLEND Ne
     4792.6410   150 BLEND Xe
     4806.0413   550 BLEND Ar
     4807.0381   500 BLEND Xe
     4810.0847   150 BLEND Ne
     4811.7768   300 BLEND Kr
     4817.6582   300 BLEND Ne
     4818.7977   150 BLEND Ne
     4821.9426   300 BLEND Ne
     4825.2031   300 BLEND Kr
     4827.3596  1000 BLEND Ne
     4827.6084   300 BLEND Ne
     4829.7318   400 BLEND Xe
     4832.0982   800 BLEND Kr
     4837.3338   500  GOOD Ne
     4843.3081   300 BLEND Xe
     4846.6332   700 BLEND Kr
     4847.8308   150 BLEND Ar
     4857.2243   150 BLEND Kr
     4879.8840   800 BLEND Ar
     4884.9392  1000 BLEND Ne
     4892.1100   500 BLEND Ne
     4916.5280   500  GOOD Xe
     4923.1721   500 BLEND Xe
     4945.6100   300 BLEND Kr
     4955.4113   150 BLEND Ne
     4957.0538  1000 BLEND Ne
     4957.1423   150 BLEND Ne
     4965.0996   200  GOOD Ar
     4971.7328   200   BAD Xe
     4972.7325   400 BLEND Xe
     4988.7881   300 BLEND Xe
     4994.9491   150 BLEND Ne
     5005.1782   500 BLEND Ne
     5022.4188   200 BLEND Kr
     5028.2992   200 BLEND Xe
     5031.3688   250 BLEND Ne
     5037.7704   500 BLEND Ne
     5044.9426   200 BLEND Xe
     5080.4040   150 BLEND Ne
     5080.6428  1000 BLEND Xe
     5086.5412   250 BLEND Kr
     5116.5218   150 BLEND Ne
     5122.2753   150 BLEND Ne
     5122.3697   150 BLEND Ne
     5122.4413   300 BLEND Xe
     5125.7504   400   BAD Kr
     5144.9558   500 BLEND Ne
     5145.0507   500 BLEND Ne
     5188.0533   300 BLEND Xe
     5188.6307   150 BLEND Ne
     5191.3924   400 BLEND Xe
     5193.1432   150 BLEND Ne
     5193.2404   150 BLEND Ne
     5203.9136   150  GOOD Ne
     5208.3377   500 BLEND Kr
     5260.4534   500 BLEND Xe
     5261.9630   500 BLEND Xe
     5292.2346  2000  GOOD Xe
     5298.2066   150  GOOD Ne
     5308.6801   200 BLEND Kr
     5309.2899   300 BLEND Xe
     5313.8887  1000 BLEND Xe
     5330.7943   600 BLEND Ne
     5333.4233   500 BLEND Kr
     5339.3517  2000 BLEND Xe
     5341.1104  1000 BLEND Ne
     5343.3005   600 BLEND Ne
     5349.2192   150  GOOD Ne
     5355.1793   150 BLEND Ne
     5355.4389   150 BLEND Ne
     5360.0274   150 BLEND Ne
     5363.2151   200 BLEND Xe
     5368.0838   200 BLEND Xe
     5372.4026   500 BLEND Xe
     5400.5780  2000 BLEND Ne
     5412.6645   250 BLEND Ne
     5418.5751   150 BLEND Ne
     5419.1697  3000 BLEND Xe
     5433.6669   250  GOOD Ne
     5438.9743   800  GOOD Xe
     5445.4625   300 BLEND Xe
     5448.5257   150 BLEND Ne
     5450.4611   200 BLEND Xe
     5460.4084   400  GOOD Xe
     5468.1862   200  GOOD Kr
     5472.6250  1000  GOOD Xe
     5525.5405   200 BLEND Xe
     5531.0890   600 BLEND Xe
     5562.2394   500 BLEND Kr
     5562.4576   150 BLEND Ne
     5562.7817   500 BLEND Ne
     5570.3032  2000 BLEND Kr
     5616.6854   300  GOOD Xe
     5656.6730   500 BLEND Ne
     5659.3937   300 BLEND Xe
     5667.5714   600 BLEND Xe
     5670.9205   150 BLEND Xe
     5681.9075   400 BLEND Kr
     5689.8302   150 BLEND Ne
     5690.3651   200   BAD Kr
     5699.6226   200 BLEND Xe
     5716.1181   200 BLEND Xe
     5718.8967   150 BLEND Ne
     5719.2393   500 BLEND Ne
     5726.9251   500  GOOD Xe
     5748.3126   500 BLEND Ne
     5751.0485   500 BLEND Xe
     5758.6664   300 BLEND Xe
     5764.4326   700 BLEND Ne
     5776.4015   300  GOOD Xe
     5804.4626   500 BLEND Ne
     5811.4204   300  GOOD Ne
     5820.1689   500 BLEND Ne
     5823.8984   300 BLEND Xe
     5824.8082   150 BLEND Xe
    # DPF checked by hand below ------------------
     5852.5007  2000  GOOD Ne
     5870.9265  3000 BLEND Kr
     5881.9079  1000 BLEND Ne
     5902.4       50 BLEND Ne  # DPF
     5905.1461   200 BLEND Xe
     5944.8464   500 BLEND Ne
     5965.4837   500 BLEND Ne
     5975.5459   600 BLEND Ne
     6030.0084  1000 BLEND Ne
     6074.3490  1000  GOOD Ne
     6096.1743   300 BLEND Ne
     6128.4609   100  GOOD Ne
     6143.0735  1000 BLEND Ne
     6163.6046  1000  GOOD Ne
     6182.1585   150 BLEND Ne
     6217.2915  1000 BLEND Ne
     6266.5050  1000  GOOD Ne
     6304.7987   100 BLEND Ne
     6334.4374  1000 BLEND Ne
     6383.0009  1000 BLEND Ne
     6402.2551  2000 BLEND Ne
     6456.2965   200  GOOD Kr
    # 6506.5366   100 BLEND Ne
    # 6532.8905   100 BLEND Ne
    # 6598.9608  1000 BLEND Ne
     6506.5366   100  GOOD Ne
     6532.8905   100  GOOD Ne
     6598.9608  1000  GOOD Ne
     6652.1002   150  FAIR Ne
     6678.2837   500 BLEND Ne
     6717.0502    70  GOOD Ne
     6728.0198   200  GOOD Xe
     6752.8400   150 BLEND Ar
     6788.7131   150 BLEND Xe
     6805.7484  1000  GOOD Xe
     6827.3225   200  FAIR Xe
     6871.2954   150 BLEND Ar  # much too weak to use
    # 6882.1674   300 BLEND Xe
     6929.4732  1000 BLEND Ne
    # 6965.4355 10000  GOOD Ar --??? adjust this, or the Ne?
     6965.442  10000  GOOD Ar
     7024.0558   500  GOOD Ne
     7030.2557   150 BLEND Ar
     7032.4185  1000 BLEND Ne
     7051.2975    70  FAIR Ne
    # 7059.1125   200  FAIR Ne # usable, but better lines nearby - DPF
     7065.1841   200 BLEND He
     7067.2225 10000 BLEND Ar
     7119.6021   500  FAIR Xe 
     7147.0456  1000 BLEND Ar
     7173.9427  1000  GOOD Ne
     7245.1707  1000  GOOD Ne
     7272.9389  2000 BLEND Ar
     7372.1207   200  FAIR Ar
    # was  7383.9834
     7383.988 10000  GOOD Ar # weak Xe at 7385.9989 - DPF says add a bit
     7438.9015   300 BLEND Ne
     7472.4415    50  FAIR Ne
     7488.8741   500 BLEND Ne
    # 7503.8704 20000  GOOD Ar
     7503.8694 20000  GOOD Ar
    # 7514.6535 15000  GOOD Ar
     7514.6545 15000  GOOD Ar
    # 7535.7767   300  GOOD Ne
     7535.7727   300  GOOD Ne
     7544.0469   100  FAIR Ne # weakest of the four - DPF
     7587.4135  1000 BLEND Kr
     7601.5466  2000 BLEND Kr # looks like a blend DPF
     7635.1073 25000  GOOD Ar
     7642.0184   500 BLEND Xe
    # 7685.2456  1000  GOOD Kr  a bit high - DPF
     7685.2400  1000  GOOD Kr
     7694.5400  1200  GOOD Kr
     7723.7620 15000 BLEND Ar
     7724.2078 10000 BLEND Ar
     7746.8286   150  FAIR Kr
     7854.8219   800  GOOD Kr
     7887.4010   300 BLEND Xe
     7913.4238   200  JUNK Kr # bad in fits
     7928.5976   180 BLEND Kr
     7943.1819   200 BLEND Ne
     7948.1763 20000  GOOD Ar
     7967.3390   500  FAIR Xe
     8006.1563 20000  GOOD Ar
     8014.7859 25000  GOOD Ar
     8059.5026  1500 BLEND Kr
     8082.4579   200 BLEND Ne
     8103.6915 20000 BLEND Ar
     8104.3633  4000 BLEND Kr
     8112.8990  6000 BLEND Kr
     8115.3103 35000 BLEND Ar
     8136.4053   300 BLEND Ne
     8190.0537  3000  GOOD Kr
     8206.3433   700 BLEND Xe
     8231.6343 10000  GOOD Xe
     8259.3780   150 BLEND Ne
     8263.2386  3000 BLEND Kr
     8264.5203 10000 BLEND Ar
     8266.0761   200 BLEND Ne
     8266.5167   500 BLEND Xe
     8280.1150  7000 BLEND Xe
     8281.0487  1500 BLEND Kr
     8298.1060  5000 BLEND Kr
     8300.3251   600 BLEND Ne
     8301.5585   150 BLEND Ne
     8346.8146  2000 BLEND Xe
    # 8365.7470   150  FAIR Ne #maybe try?
     8365.7470   150  GOOD Ne #maybe try?
     8376.3599   200 BLEND Ne
     8377.6050   800 BLEND Ne
     8408.2067 15000 BLEND Ar
     8418.4257   400 BLEND Ne
     8424.6452 20000  GOOD Ar # sat? pulled by neighbor?
     8463.3556   150  FAIR Ne # too weak? - DPF
     8495.3578   500  GOOD Ne
     8508.8681  3000  GOOD Kr
     8521.4386 15000  GOOD Ar
     8591.2562   400  GOOD Ne
     8634.6443   600 BLEND Ne
     8647.0383   300 BLEND Ne
     8648.5417   250 BLEND Xe
     8654.3802  1500 BLEND Ne
     8655.5195   400 BLEND Ne
     8667.9403  4500 BLEND Ar
     8679.4896   500 BLEND Ne
     8681.9182   500 BLEND Ne
     8704.1086   200  GOOD Ne
    # 8739.3867   300  GOOD Xe This must be too high
     8739.375    300  GOOD Xe
     8771.6530   400 BLEND Ne
     8776.7444  6000 BLEND Kr
     8778.7294   150 BLEND Ne
     8780.6176  1200 BLEND Ne
     8783.7499  1000 BLEND Ne
     8819.4047  5000  GOOD Xe
     8853.8631   700 BLEND Ne
     8862.3129   300 BLEND Xe
     8865.7514   500 BLEND Ne
     8919.4966   300  JUNK Ne  # trouble maker
     8928.6876  2000 BLEND Kr
     8930.8240   200 BLEND Xe
    # 8952.2481  1000  JUNK Xe  # Maybe bad?  try this one again
     8952.2481  1000  GOOD Xe  # Maybe this is OK - DPF
     8987.5684   200 BLEND Xe
     8988.5521   200 BLEND Ne
    # 9045.4425   400  JUNK Xe  good line!
     9045.4425   400  GOOD Xe # try this?
     9122.9622 35000  GOOD Ar
     9148.6666   600  FAIR Ne # weaker than neighbors - DPF
     9162.6403   500  GOOD Xe
     9194.6325   550  FAIR Ar
     9201.7539   600  JUNK Ne # trouble maker
     9220.0530   400 BLEND Ne
     9221.5749   200 BLEND Ne
     9224.4923 15000 BLEND Ar
     9226.6850   200 BLEND Ne
     9291.5258   400 BLEND Ar
    # checked by hand above here - DPF
     9300.8471   600  JUNK Ne # never part of a good fit...
     9310.5782   150 BLEND Ne
     9313.9669   300 BLEND Ne
     9320.9867   200   BAD Kr
     9326.5011   600  JUNK Ne
     9354.2126  1600  GOOD Ar
     9361.9455   300 BLEND Kr
     9373.3019   200 BLEND Ne
     9402.8142   200   BAD Kr
     9425.3727   500  GOOD Ne
     9459.2032   300  GOOD Ne
     9470.9255   200   BAD Kr
     9486.6755   500  GOOD Ne
     9513.3738   200  GOOD Xe
     9534.1563   500  GOOD Ne
     9547.3983   300  GOOD Ne
     9577.0063   120 BLEND Ne
     9577.5162   500 BLEND Kr
     9605.7884   500   BAD Kr
     9619.6046   400   BAD Kr
     9657.7791 25000  GOOD Ar
     9663.3326   200 BLEND Kr
     9665.4126  1000 BLEND Ne
     9685.3165   150  GOOD Xe
     9711.5893   200   BAD Kr
     9751.7503  2000  GOOD Kr
     9784.4943  4500  GOOD Ar
     9799.6951  2000 BLEND Xe
     9803.1341   500 BLEND Kr
     9856.3065   500  GOOD Kr
     9923.1811  3000  GOOD Xe
    10052.0557   180  GOOD Ar
    10221.4491  1000  GOOD Kr
    10470.0427  1600 BLEND Ar
    10506.4897   180  GOOD Ar
    10562.3968   200  GOOD Ne
    10673.5548   200  GOOD Ar
    10798.0313   150  GOOD Ne
    10829.0794   300 BLEND He
    10830.2384  1000 BLEND He
    10830.3281  2000 BLEND He
    10844.4655   200  GOOD Ne
    ''')

    return arc_data
