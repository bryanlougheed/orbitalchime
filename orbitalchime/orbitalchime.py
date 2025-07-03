import numpy as np
from numba import njit, prange
import pandas as pd
import os
import warnings

def getlaskar2004(option=1, timeslice=(-np.inf, np.inf)):
    """
    tka, ecc, obl, lpe = getlaskar2004(option=1, timeslice=(-np.inf, np.inf))

    Open data files for the Laskar2004 solution (Laskar et al., 2004).
    Downloaded from http://vo.imcce.fr/insola/earth/online/earth/La2004/index.html

    Parameters
    ----------
    option : integer
        option = 1, 51 Ma to 0 Ma
        option = 2, 0 Ma to 21 Ma in the future
        option = 3, 101 Ma to 0 Ma
        option = 4, 249 Ma to 0 Ma
        option = 5, 51 Ma to 21 Ma in the future (concatenate options 1 & 2)

    timeslice : array-like
        Contains one or two values (in ka before 2000 CE, negative values = future from 2000 CE)
        If one value, a single time interval. If two values, all time intervals between the two values.
        If not given, all time slices in the dataset will be returned.

    Returns
    -------
    tka, ecc, obl, lpe

    tka: ndarray
        time in ka before year 2000 CE (negative values = future from 2000 CE)
    ecc: ndarray
        eccentricity (dimensionless: https://en.wikipedia.org/wiki/Orbital_eccentricity)
    obl: ndarray
        obliquity (radians)
    lpe: ndarray
        longitude of perihelion from moving equinox (radians, heliocentric)

    Information
    -----------
    Script originally written in Matlab by B.C. Lougheed in 2020.
    Ported to python/pandas/numpy by B.C. Lougheed in Oct. 2024.

    Reference for the imported data:
    Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A.C.M., Levrard, B., 2004.
    "A long-term numerical solution for the insolation quantities of the Earth. "
    A&A 428, 261-285. https://doi.org/10.1051/0004-6361:20041335
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dirloc = file_dir+'/laskar_et_al/'

    def scinot(val): # because Laskar et al output D (from fortran) instead of E for overflow
        return float(val.replace('D', 'E'))
 
    if option == 1:
        d = pd.read_csv(dirloc+'INSOLN.LA2004.BTL.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
    elif option == 2:
        d = pd.read_csv(dirloc+'INSOLP.LA2004.BTL.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
    elif option == 3:
        d = pd.read_csv(dirloc+'INSOLN.LA2004.BTL.100.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
    elif option == 4:
        d = pd.read_csv(dirloc+'INSOLN.LA2004.BTL.250.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
    elif option == 5:
        d1 = pd.read_csv(dirloc+'INSOLN.LA2004.BTL.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
        d2 = pd.read_csv(dirloc+'INSOLP.LA2004.BTL.ASC', sep=r'\s+', header=None, skiprows=3, converters={1: scinot, 2: scinot, 3: scinot}).values
        d = np.vstack((d1[1:], d2))
    
    # prep the data
    d[:, 0] = (d[:, 0] * -1) # geo style: make past positive, future negative
    d = d[d[:, 0].argsort()][::-1]  # sort rows by the first column in descending order

    # get the requested time slice
    timeslice = np.array([timeslice])
    d = d[(d[:, 0] >= np.min(timeslice)) & (d[:, 0] <= np.max(timeslice))]

    # return np arrays
    tka = np.array(d[:, 0])
    ecc = np.array(d[:, 1])
    obl = np.array(d[:, 2])
    lpe = np.array(d[:, 3])

    return tka, ecc, obl, lpe

def getlaskar2010(option=1, timeslice=(-np.inf, np.inf)):
    """
    tka, ecc = getlaskar2010(option)

    Open Laskar2010 eccentricity solution data files. Useful for looking at eccentricity > 30 Ma.
    Downloaded from http://vo.imcce.fr/insola/earth/online/earth/La2010/index.html

    Parameters
    ----------
    option: integer
        option = 1, La2010a (solution a)
        option = 2, La2010b (solution b)
        option = 3, La2010c (solution c)
        option = 4, La2010d (solution d)
    timeslice : array-like, containing one or two values
        If one value, a single time interval. If two values, minimum and maximum time interval (in ka before 2000 CE)
        If not given, all time slices in the dataset will be returned.

    Returns
    -------
    tka, ecc

    tka: ndarray
        time in ka before year 2000 CE (negative years = future from 2000 CE)
    ecc: ndarray
        eccentricity (dimensionless: https://en.wikipedia.org/wiki/Orbital_eccentricity)

    Information
    -----------
    Script originally written in Matlab 2019a by B.C. Lougheed in 2020.
    Ported to python/pandas/numpy by B.C. Lougheed in Oct. 2024.
    Python 3.12.4, pandas 2.2.2, numpy 1.26.4.

    Reference for the imported data:
    Laskar, J., Fienga, A., Gastineau, M., Manche, H., 2011.
    "La2010: a new orbital solution for the long-term motion of the Earth."
    A&A 532, A89. https://doi.org/10.1051/0004-6361/201116836
    """

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dirloc = file_dir+'/laskar_et_al/'
    
    if option == 1:
        d = pd.read_csv(dirloc+'La2010a_ecc3L.dat', sep=r'\s+', header=None).values
    elif option == 2:
        d = pd.read_csv(dirloc+'La2010b_ecc3L.dat', sep=r'\s+', header=None).values
    elif option == 3:
        d = pd.read_csv(dirloc+'La2010c_ecc3L.dat', sep=r'\s+', header=None).values
    elif option == 4:
        d = pd.read_csv(dirloc+'La2010d_ecc3L.dat', sep=r'\s+', header=None).values

    d[:, 0] = (d[:, 0] * -1) - (50 / 1000)  # ka 1950
    d = d[d[:, 0].argsort()][::-1]  # sort rows by the first column in descending order

    # prep the data
    d[:, 0] = (d[:, 0] * -1) # geo style: make past positive, future negative
    d = d[d[:, 0].argsort()][::-1]  # sort rows by the first column in descending order

    # get the requested time slice
    timeslice = np.array([timeslice]) 
    d = d[(d[:, 0] >= np.min(timeslice)) & (d[:, 0] <= np.max(timeslice))]

    tka = np.array(d[:, 0])
    ecc = np.array(d[:, 1])

    return tka, ecc


@njit(parallel=True)
def keplerjitloop(M, ecc, floatpp):
    # define these here as constant
    # (instead of repeat calls of np.pi in loop)
    kvartpi = np.pi/4
    halvpi = np.pi/2
    pipi = 2*np.pi

    n = M.size
    E = np.empty(n)
    precision = np.empty(n)

    for i in prange(n):
        Mi = M[i]
        ecci = ecc[i]
        sign_Mi = 1.0
        if Mi < 0:
            Mi = -Mi
            sign_Mi = -1.0

        Mi = (Mi % pipi) * sign_Mi
        if Mi < 0:
            Mi += pipi

        F = 1.0
        if Mi > np.pi:
            F = -1.0
            Mi = pipi - Mi

        Eo = halvpi
        D = kvartpi
        iters = int((floatpp / 4) * 3.3 + 5)

        for j in range(iters):
            if j == iters - 1:
                Eoo = Eo
            M1 = Eo - ecci * np.sin(Eo)
            Eo += D * np.sign(Mi - M1)
            D *= 0.5

        E[i] = Eo * F
        precision[i] = abs(Eo - Eoo)

    return E, precision

def solvekeplerE(M, ecc, floatpp=64, mode='numba'):
    """
    E, precision = solvekeplerE(M, ecc, floatpp=64, mode='auto')

    Solves Kepler equation for E to within machine precision by
    using Sinnot (1985) binary search method.

    Suitable for eccentricity values between 0 and 0.98.

    Parameters
    ----------
    M : ndarray
        Mean anomaly (radians)
    ecc : ndarray
        Eccentricity of the ellipse (ratio)
    floatpp : integer (optional)
        Floating point precision you are using (default = 64)
    mode : string (optional)
        'auto' (default), 'numba' or 'numpy'
        'auto' will automatically select 'numba' for large inputs

    Returns
    -------
    E : ndarray
        Eccentric anomaly (radians)
    precision : ndarray
        The precision on the eccentric anomaly (difference between the final two loop iterations).

    Information
    -----------
    Roger Sinnott (1985) BASIC script, as suggested by Meeus (1998).
    Solves Kepler equation for E, using binary search, to within computer precision.
    BASIC ported to Matlab by Tiho Kostadinov (Kostadinov and Gilb, 2014). 
    Matlab version vectorised to handle array input by Bryan Lougheed (Lougheed, 2022). 
    Subsequently ported to python/numpy in October 2024 by Bryan Lougheed.
    Numba-optimised run mode added July 2025 by Bryan Lougheed.

    References
    ----------
    R.W. Sinnott (1985), "A computer assault on Kepler's equation", Sky and Telescope, vol. 70, page 159.
    J. Meeus (1998). Chapter 30 in Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia.
    Kostadinov and Gilb, (2014): doi:10.5194/gmd-7-1051-2014.
    B.C. Lougheed (2022), doi:10.5334/oq.100.
    """
    if mode == 'auto':
        total_size = np.broadcast(M, ecc).size
        if total_size > 450000:
            mode = 'numba'
        else:
            mode = 'numpy'

    if mode == 'numpy':
        # pure numpy vectorised version, runs on single thread :(
        M, ecc = np.broadcast_arrays(M, ecc) # not necessary in matlab, needed here in numpy

        F = np.sign(M)
        M = np.abs(M) / (2*np.pi)
        M = (M-np.floor(M)) * 2*np.pi * F
        M[M<0] += 2*np.pi  # put in same relative orbit
        F = np.ones_like(M)

        mask = M>np.pi
        F[mask] = -1
        M[mask] = 2*np.pi - M[mask]  # inbound
        
        Eo = np.full_like(ecc, np.pi/2)
        D = np.full_like(ecc, np.pi/4)
        # Converging loop
        # Sinnot says number of iterations is 3.30 * significant figures of system. 
        # Matlab double has 16 digit precision, so 16*3.30=53. Let's use 58
        # np int64 is the same
        iters = np.ceil((floatpp/4)*3.3+5)
        for i in prange(int(iters)): 
            if i == int(iters-1):
                Eoo = np.copy(Eo)
            M1 = Eo - ecc * np.sin(Eo)
            Eo += D * np.sign(M - M1)
            D /= 2

        E = Eo * F
        precision = np.abs(Eo - Eoo)

    elif mode == 'numba':
        # numpy+numba jit linear version optimised for multithreading
        # Ensure inputs are arrays and broadcasted to same shape
        M, ecc = np.broadcast_arrays(np.asarray(M), np.asarray(ecc))
        M.setflags(write=True)
        ecc.setflags(write=True)
        M_flat = M.ravel()
        ecc_flat = ecc.ravel()

        # pass through keplerjitlooop function, which is defined
        # outside this function so it is compiled only once
        E_flat, prec_flat = keplerjitloop(M_flat, ecc_flat, floatpp)

        # put back into original broadcast shape
        E = E_flat.reshape(M.shape)
        precision = prec_flat.reshape(M.shape)
        
            
    else:
        raise ValueError("mode must be set to 'auto','numba' or 'numpy'")
        
    return E, precision


def sollon2time(sollon, ecc, lpe, tottime=365.24, obl=None):
    """
    time, eot = sollon2time(sollon, ecc, lpe, tottime=365.24, obl=None)

    Given a particular eccentricity and longitude of perihelion, get time of tropical year
    associated with a particular geocentric solar longitude, i.e. by accounting for 
    conservation of angular momentum during orbit (Kepler 2nd Law).

    Parameters
    ----------
    sollon : array-like
        Keplerian geocentric solar longitude in radians ('v' relative to NH spring equinox)
        Either 1 value (used as constant if other inputs are vector), or a vector of values.
    ecc : array-like
        Eccentricity (e.g., from Laskar et al.)
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        omega (i.e., relative to NH autumn equinox) in radians.
    tottime : float
        Total time in the year, single value, any time unit you want. Default value is 365.24.
    obl : array-like, optional
        Obliquity in radians (e.g., from Laskar et al.) for calculating the equation of time (eot)

    Returns
    -------
    time, eot

    time : ndarray
        Time interval of tropical year (where 0 is boreal spring equinox).
    eot : ndarray
        Equation of time (minutes). Returns empty if obl not supplied.

    Information
    -----------
    Bryan Lougheed, June 2020, Matlab 2019a
    Updated April 2023 to include eot.
    Converted to python/numpy October 2024 by Bryan Lougheed.
    Python 3.12.4, numpy 1.26.4.
    
    See following for background, as well as comments in the script:
    J. Meeus, (1998). Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia. (specifically Chapter 30).
    Also: https://dr-phill-edwards.eu/Astrophysics/EOT.html (for equation of time)
    """

    # Change lpe from heliocentric to geocentric
    omega = lpe + np.pi
    omega[omega >= 2*np.pi] -= 2*np.pi  # wrap to 360

    # Get day of anchor day (dz) relative to perihelion
    vz = 2*np.pi - omega  # v of spring equinox relative to perihelion
    vz[vz > 2*np.pi] -= 2*np.pi
    Ez = 2 * np.arctan(np.tan(vz / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # Meeus (1998) page 195, solve for E
    Mz = Ez - ecc * np.sin(Ez)  # Meeus page 195, solve for M (Kepler equation). M is the circular orbit equivalent of v
    Mz[Mz < 0] = np.pi + (np.pi - Mz[Mz < 0] * -1)  # inbound to perihelion
    dz = Mz / (2*np.pi) * tottime

    # Get day of target day (dx) relative to perihelion
    vx = vz + sollon
    vx[vx > 2*np.pi] -= 2 * np.pi
    Ex = 2 * np.arctan(np.tan(vx / 2) * np.sqrt((1 - ecc) / (1 + ecc)))  # Meeus (1998) page 195, solve for E
    Mx = Ex - ecc * np.sin(Ex)  # Solve for M (Kepler equation)
    Mx[Mx<0] = np.pi + (np.pi - Mx[Mx < 0] * -1)  # inbound to perihelion, (probably not necessary)
    dx = Mx / (2*np.pi) * tottime

    # Get day of target day (dx) relative to day of anchor day (dz)
    dx[dx<dz] += tottime  # for dz in next orbital period relative to perihelion, keep in same orbital period relative to NH spring equinox
    time = dx - dz

    # Eliminate rounding errors at zero
    sollon, time = np.broadcast_arrays(sollon, time)
    time[sollon == 0] = 0
    time[sollon == 2*np.pi] = 0

    # Calculate equation of time if obl is supplied
    # https://dr-phill-edwards.eu/Astrophysics/EOT.html (explains it very nicely) 
    if obl is not None:
        # eccentricity component
        dtecc = np.rad2deg(Mx-vx) * 4  # four minutes per degree (24 hrs * 60 mins / 360 degrees )
        # obliquity component
        alpha = np.arctan2(np.sin(sollon) * np.cos(obl), np.cos(sollon))
        alpha[alpha<0] += 2*np.pi
        dtobl = np.rad2deg(sollon-alpha) * 4 # same here
        # total EOT, time in minutes
        eot = dtecc + dtobl
    else:
        eot = np.array([])

    return time, eot

def time2sollon(time, ecc, lpe, tottime=365.24, obl=None, floatpp=64):
    """
    sollon, eot = time2sollon(time, ecc, lpe, tottime=365.24, obl=None, floatpp=64)

    Given a particular eccentricity and longitude of perihelion, get geocentric solar longitude 
    associated with a particular time of the tropical year i.e. by accounting for 
    conservation of angular momentum during orbit (Kepler 2nd Law).

    Parameters
    ----------
    time : ndarray
        Time interval of tropical year (where interval 0 is boreal spring equinox). Either on value, or vector of values.
    ecc : ndarray
        Eccentricity (e.g., from Laskar et al.)
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        omega (i.e., relative to NH autumn equinox) in radians.
    tottime : float
        Total time in the year corresponding to 'time', single value, any time unit you want. Default value is 365.24.
    obl : ndarray, optional
        Obliquity in radians (e.g., from Laskar et al.) for calculating the equation of time (eot)
    floatpp : integer, optional
        Floating point precision you are using (default = 64 bit)

    Input can be vectorised in various ways, but double check output.

    Returns
    -------
    sollon : np.array
        Keplerian geocentric solar longitude in radians ('lambda', i.e. 'v' relative to boreal spring equinox) 
    eot : np.array
        Equation of time (minutes). Returns empty if obl not supplied.

    Info
    ----
    B.C. Lougheed, June 2020, Matlab 2019a
    Updated April 2023 to include eot.
    Ported to python/numpy in October 2024 by B.C. Lougheed.
    Python 3.12.4, numpy 1.26.4.

    See following for background, as well as comments in the script:
    R.W. Sinnott (1985), "A computer assault on Kepler's equation." Sky and Telescope, vol. 70, page 159.
    Meeus, J., (1998). Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia. (specifically Chapter 30).
    B.C. Lougheed (2022), doi:10.5334/oq.100.
    https://dr-phill-edwards.eu/Astrophysics/EOT.html (for equation of time)
    """
    # convert input to numpy for speed and compatibility
    tottime = np.array([tottime])
    time = time.reshape(-1,1)

    # change lpe from heliocentric to geocentric
    omegabar = np.array(lpe + np.pi) # np.array needed for when doing only one timeslice
    omegabar[omegabar >= 2*np.pi] = omegabar[omegabar >= 2*np.pi] - 2*np.pi

    # NH spring equinox relative to perihelion
    veq = 2*np.pi - omegabar
    Eeq = 2 * np.arctan(np.tan(veq/2) * np.sqrt((1-ecc) / (1+ecc)))
    Meq = np.array(Eeq - ecc * np.sin(Eeq)) # as previous comment
    Meq[Meq<0] = np.pi + (np.pi - Meq[Meq<0] * -1)
    deq = Meq / (2*np.pi) * tottime

    # v of target (x) v relative to perihelion
    deq, time = np.broadcast_arrays(deq,time)
    dx = deq + time
    Mx = (dx / tottime) * 2*np.pi
    Ex, _ = solvekeplerE(Mx, ecc, floatpp=floatpp)  # Get Ex by solving Kepler equation
    vx = 2 * np.arctan(np.tan(Ex/2) * np.sqrt((1+ecc) / (1-ecc)))
    vx[vx<0] = np.pi + (np.pi - vx[vx<0] * -1) # incoming
    
    # target day's v relative to NH spring equinox v
    sollon = vx - veq

    # eliminate rounding errors at 0
    sollon, time = np.broadcast_arrays(sollon, time)
    sollon[time == 0] = 0
    sollon[time == tottime] = 0
    sollon = np.array(sollon)

    if obl is not None:
        # eccentricity component
        dtecc = np.rad2deg(Mx - vx) * 4
        # obliquity component
        alpha = np.arctan2(np.sin(sollon) * np.cos(obl), np.cos(sollon))
        alpha[alpha<0] += 2*np.pi
        dtobl = np.rad2deg(sollon - alpha) * 4
        # total EOT, time in minutes
        eot = dtecc + dtobl
    else:
        eot = np.array([])

    sollon[sollon<0] += 2*np.pi
    sollon[sollon>2*np.pi] -= 2*np.pi

    return sollon, eot

def geographiclat(gclat, angles='rad'):
    """
    gplat = geographiclat(gclat, angles='rad')

    Convert geocentric latitude into geographic latitude
    assuming the WGS84 spheroid.

    Parameters
    ----------
    gclat : array-like
        Geocentric latitude.
    angles : string (optional)
        'rad' (default) or 'deg'. 
        Specify if gclat is in degrees or radians.

    Returns
    -------
    gplat : ndarray
        Geographic latitude in radians.

    Bryan Lougheed, February 2025.
    """
    gclat = np.array(gclat)

    if angles == 'rad':
        pass
    elif angles == 'deg':
        gclat = np.deg2rad(gclat)
    else:
        raise Exception("'angles' parameter should be set to either 'deg' or 'rad'")

    # calculate geographic latitude from geocentric latitude
    f = 1 / 298.257223563  # wgs84 flattening value
    re = 6378137.0  # wgs84 equatorial radius (metres)
    rp = re * (1 - f)  # calculate polar radius
    gplat = np.arctan((re / rp)**2 * np.tan(gclat))

    return gplat

def dailymeanwm2(lat, sollon, ecc, obl, lpe, con=1361, earthshape='sphere'):
    """
    irr, dayhrs, rx, tsi = dailymeanwm2(lat, sollon, ecc, obl, lpe, con=1361, earthshape='sphere')

    Calculate 24-hr mean irradiance (W/m²) at top of atmosphere and also length of daytime (in hours), 
    total solar irradiance (TSI; in W/m²) and distance from sun (in AU).

    Parameters
    ----------
    lat : array-like
        Geocentric latitude (plus for N, minus for S) on Earth, in radians.
    sollon : array-like
        Geocentric solar longitude (lambda), in radians.
    ecc : array-like
        Eccentricity. Numerical value(s). 1D array.
    obl : array-like
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        omega (i.e., relative to NH autumn equinox) in radians.
    con : float or array-like, optional
        Solar constant in W/m². Single numerical value or 1D array. Default is 1361 W/m².
    earthshape : str, optional
        Shape of Earth, enter string 'sphere' or 'wgs84' (default is 'sphere').

    Returns
    -------
    irr, dayhrs, tsi, rx

    irr : ndarray
        Calculated mean daily (24 hr) irradiance (W/m²) at top of atmosphere. Array same size as ecc, obl and lpe.
    dayhrs : ndarray
        Hours of daylight. Array same size as ecc, obl and lpe.
    rx : ndarray
        Distance from Sun (AU). Insensitive to latitude, obliquity or earthshape.
    tsi : ndarray
        Calculated mean daily irradiance at top of atmosphere assuming 90 degree angle of incidence, W/m².
        Insensitive to latitude, obliquity or earthshape. Array same size as ecc, obl and lpe.

    Info
    ----
    B.C. Lougheed, May 2020, Matlab 2019a
    Updated to include Earth's oblateness Sep. 2020
    Ported to python/numpy Oct. 2024 by B.C. Lougheed
    Python 3.12.4, numpy 1.26.4.

    irr in Wm² based on equations in Berger (1978).
    Berger, A.L. 1978. "Long-Term Variations of Daily Insolation and Quaternary Climatic Changes."
    J. Atmos. Sci., 35: 2362-2367.
    
    I added ability to take oblateness of Earth into account, validated against Van Hemelrijck (1983) solution. (see comments in script)
    
    I added daylight hours output following sunrise equation: 
    https://en.wikipedia.org/wiki/Sunrise_equation
    
    I added tsi by calculating distance from sun following Meeus (1998).
    Meeus, J., (1998). Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia. (specifically Chapter 30)    
    """
    # numpy everything just to be sure
    lat = np.array(lat)
    sollon = np.array(sollon)
    ecc = np.array(ecc)
    obl = np.array(obl)
    lpe = np.array(lpe)
    con = np.array(con)
    
    # Check for NaN in input
    checklist = {"lat": lat, "sollon": sollon, "ecc": ecc, "obl": obl, "lpe": lpe, "con": con}
    for name, array in checklist.items():
        if np.isnan(array).any():
            warnings.warn(f"Inputted {name} contains NaN, which could cause erroneous calculations", UserWarning)

    ### Calculate rx and tsi
    omegabar = np.array(lpe + np.pi)  # add 180 degrees. (heliocentric to geocentric)
    omegabar[omegabar >= 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    veq = 2*np.pi - omegabar  # v (true anomaly) of spring equinox relative to perihelion
    vx = np.array(veq + sollon)  # v (true anomaly) of inputted sollon relative to perihelion
    vx[vx > 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    rx = (1 - ecc**2) / (1 + ecc * np.cos(vx))  # Distance from Sun in AU, Eq. 30.3 in Meeus (1998)
    tsi = con * (1 / rx)**2 # Total solar irradiance at distance rx

    if earthshape == 'sphere' or earthshape == 'wgs84':

        if earthshape == 'wgs84':
            lat = geographiclat(lat)
 
        # Declination angle of the sun
        # https://en.wikipedia.org/wiki/Position_of_the_Sun
        dsun = np.arcsin(np.sin(obl) * np.sin(sollon))

        #if earthshape == 'wgs84':
        #    dsun = geographiclat(dsun)

        # Hour angle at sunrise/sunset
        # https://en.wikipedia.org/wiki/Sunrise_equation
        # Invalid input to arccos caused by polar day or polar night will return NaN. 
        # NaN output warning supressed here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hangle = np.arccos(-np.tan(lat) * np.tan(dsun))
        
        # Put hangle pi for polar day and zero for polar night (the NaN values in hangle)
        # If lat*dsun is >0 then polar region and an subsolar point are both in same hemisphere --> polar day
        # https://en.wikipedia.org/wiki/Hour_angle
        hangle[np.logical_and(np.isnan(hangle), lat*dsun > 0)] = np.pi  # polar day
        hangle[np.logical_and(np.isnan(hangle), lat*dsun <= 0)] = 0    # polar night
        
        # Hours of daylight (https://en.wikipedia.org/wiki/Sunrise_equation)
        dayhrs = np.abs(hangle - hangle*-1) / (2*np.pi / 24)

        # 24 hr mean irradiance: Berger (1978) eq (10)
        # Omit the 86.4 factor included in Berger's eq (10), which is a conversion factor for W to kJ (60*60*24/1000)
        # You could also use Berger eq (8) and eq (9) for polar day and night
        # However, here, hangle = pi is inputted for (polar day) and hangle = 0 for polar night into Eq (10),
        # thus facilitating vecotorised programming. It gives the same irr output for polar day and night as following
        # Berger eqs (8) and (9) separately for polar day and night
        irr = (tsi/np.pi) * ( hangle * np.sin(lat) * np.sin(dsun) + np.cos(lat) * np.cos(dsun) * np.sin(hangle))

    elif earthshape == 'vanh':
     
        # Van Hemelrijck (1983) extended method for oblate Earth
        # produces exact same output as inputting geographic latitude into 
        # Berger equation for spherical Earth, which seems easier..

        # Declination angle of the sun
        # https://en.wikipedia.org/wiki/Position_of_the_Sun
        dsun = np.arcsin(np.sin(obl) * np.sin(sollon))
        # Hour angle at sunrise/sunset
        # https://en.wikipedia.org/wiki/Sunrise_equation
        # use geogrpaphic lat following Van Hemelrijck
        gglat = geographiclat(lat)
        hangle = np.arccos(np.tan(gglat) * -np.tan(dsun))

        # https://en.wikipedia.org/wiki/Hour_angle
        hangle[np.logical_and(np.isnan(hangle), lat*dsun > 0)] = np.pi  # polar day
        hangle[np.logical_and(np.isnan(hangle), lat*dsun <= 0)] = 0    # polar night
        
        # Hours of daylight (https://en.wikipedia.org/wiki/Sunrise_equation)
        dayhrs = np.abs(hangle - hangle*-1) / (2*np.pi / 24)
      
        f = 1 / 298.257223563 # wgs84 flattening
        vangle = np.arctan((1 - f)**-2 * np.tan(lat)) - lat  # Van Hemelrijck (1983) eq. 9, f is wgs84 flattening
        irr = (tsi * 1 / np.pi) * (np.cos(vangle) * (hangle * np.sin(lat) * np.sin(dsun) + np.sin(hangle) * np.cos(lat) * np.cos(dsun)) + np.sin(vangle) * (-np.tan(lat) * (hangle * np.sin(lat) * np.sin(dsun) + np.sin(hangle) * np.cos(lat) * np.cos(dsun)) + hangle * np.sin(dsun) / np.cos(lat)))
           
    else:
        raise ValueError('earthshape '+earthshape+' unrecognised')


    return irr, dayhrs, rx, tsi

def intradaywm2(lat, ecc, obl, lpe, dayint, daysinyear=365.242, con=1361.0):
    """
    irr, elev, msdhr, lashr, eot = intradaywm2(lat, ecc, obl, lpe, dayint, daysinyear=365.242, con=1361.0)

    Calculate intraday irradiance (W/m²) for a particular  latitude and orbital configuration. 
    Calculations assume a longitude where northern spring equinox occurs at day 0.0 (i.e., at exactly
    local midnight on the first day of the tropical year). Script takes equation of time into account.

    Parameters
    ----------
    lat : ndarray
        geocentric latitude in radians (positive for north, negative for south)
    ecc : ndarray
        eccentricity of the ellipse, ratio (from, e.g., Laskar et al.)
    obl : ndarray
        obliquity in radians (from, e.g., Laskar et al.)
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        omega (i.e., relative to NH autumn equinox) in radians.
    dayint : ndarray
        The mean solar day interval(s) to be analysed, in day decimals (i.e. 0.5 for midday on day 0).
        Note that, due to the equation of time, 0.5 would not necessarily exactly correspond to local solar noon.
        Calculations will assume that dayint 0.0 (midnight on day zero) corresponds to the northern hemisphere
        spring equinox.
    daysinyear : float
        The number of mean solar days in the year, default is 365.242
    con : float
        solar constant in W/m², default is 1361 W/m²

    Returns
    -------
    irr, elev, msdhr, lashr, eot

    irr : ndarray
        array of W/m² for every mean solar day interval calculated
    elev : ndarray
        solar elevation (radians), same dimension as irr
    msdhr : ndarray
        time of the the mean solar day (0-24, in hours), same dimension as irr
        Can be best thought of as "clock hours".
    lashr : ndarray
        local apparent solar hour (0-24), same dimension as irr
        Can be best thought of as "sun dial hours".
    eot : ndarray
        Equation of time (minutes). Returns empty if obl not supplied.

    Bryan Lougheed, April 2023, Matlab 2019a
    Ported to python/numpy by Bryan Lougheed, Oct. 2024
    Python 3.12.4, numpy 1.26.4.
    """

    # convert some input to numpy for speed (might not be necessary)
    con = np.array([con])

    # Calculate Earth's solar longitude (i.e., lambda) and equation of time for each day fraction
    sunlon, eot = time2sollon(dayint, ecc, lpe, daysinyear, obl)

    # mean solar day length in hours (placeholder for future development)
    # has not been properly implemented/tested yet for day lengths other than 24
    msdlen = 24 

    # Create vector of mean solar day hours
    msdhr = (dayint - np.floor(dayint)) * msdlen

    # Get local apparent solar hour (correct for eot)
    msdhr = msdhr.reshape(-1,1)
    eot, msdhr = np.broadcast_arrays(eot,msdhr)
    lashr = (eot/60) + msdhr # /60, mins -> hrs
    lashr[lashr<0] += msdlen
    lashr[lashr>msdlen] -= msdlen

    # Declination of the sun
    dsun = np.arcsin(np.sin(obl) * np.sin(sunlon))

    # Local hour angle (-pi to +pi radians, midday = 0 radians)
    hangle = (2*np.pi / msdlen) * (lashr - msdlen/2)

    # Solar elevation
    elev = np.arcsin(np.sin(dsun) * np.sin(lat) + np.cos(dsun) * np.cos(lat) * np.cos(hangle))

    # Calculate distance from Sun in AU
    omegabar = lpe + np.pi
    veq = 2*np.pi - omegabar  # v (true anomaly) of NH spring equinox relative to perihelion
    vx = veq + sunlon  # v (true anomaly) of inputted sunlon relative to perihelion
    vx[vx > 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    rx = (1 - ecc**2) / (1 + ecc * np.cos(vx))  # Eq. 30.3 in Meeus (1998)

    # Calculate tsi as function of con relative to 1 AU
    tsi = con * (1/rx)**2

    # Calculate TOA W/m2, i.e. the vertical component of TSI W/m2
    irr = tsi * np.sin(elev)
    irr[irr < 0] = 0  # sun under horizon, night time

    return irr, elev, msdhr, lashr, eot

def thresholdjm2(thresh, lat, ecc, obl, lpe, con=1361, timeres=0.01, tottime=365.24, earthshape='sphere'):
    """
    intirr, ndays = thresholdjm2(thresh, lat, ecc, obl, lpe, con=1361, timeres=0.01, tottime=365.24, earthshape='sphere')

    Calculate integrated irradiation (J/m²) at top of atmosphere for all day intervals 
    exceeding a certain threshold in mean daily irradiance (W/m²).
    Can be used to emulate analysis by, e.g., Huybers (2006; 10.1126/science.1125249)
    
    Parameters
    ----------
    thresh : float or array-like
        Threshold value (W/m2). Single value, or vector of values.
    lat : float
        Geocentric latitude (in deg. N, negative for S) on Earth. Single value.
    con : float or array-like, optional
        Solar constant. Single numerical value or 1D array, W/m2. Default is 1361.
    dayres : float, optional
        Day resolution for the integration. Default is 0.01.
    ecc : array-like
        Eccentricity. Numerical value(s). 1D array.
    obl : array-like
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        omega (i.e., relative to NH autumn equinox) in radians.
    earthshape : str (optional)
        Shape of Earth, 'sphere' (default) or 'wgs84'.

    Returns
    -------
    intirr : ndarray
        Integrated irradiation at top of atmosphere for days exceeding thresh. J/m2. Array same dimensions as ecc, obl, and lpe.
    ndays : ndarray
        Time (in days) exceeding thresh. Same dimensions as intirr.
    """

    timerange = np.arange(0, tottime, timeres)
    intirr = np.full_like(ecc, np.nan)
    ndays = np.full_like(ecc, np.nan)

    # this needs to be vectorised
    for i in range(len(ecc)):
        sollons, _ = time2sollon(timerange, ecc[i], lpe[i], tottime)
        irrs, _, _, _ = dailymeanwm2(lat, sollons, ecc[i], obl[i], lpe[i], con=1361, earthshape='sphere')
        ndays[i] = np.sum(irrs >= thresh) * timeres
        intirr[i] = np.mean(irrs[irrs >= thresh]) * (ndays[i] * 24 * 60 * 60)  # W/m2 to J/m2

    return intirr, ndays

def areaquad(lat1, lat2, lon1, lon2, shape='sphere', angles='rad'):
    """
    aq = areaquad(lat1, lat2, lon1, lon2, shape='sphere', angles='rad')
    
    Calculate the surface area of a lat/lon bounding box on Earth.

    Inputs lat1, lat2, lon1 and lon2 must all be of same shape.

    Parameters
    ----------
    lat1 : array-like
        A bounding geocentric latitude.
    lat2 : array-like
        The other bounding geocentric latitude.
    lon1 : array-like
        A bounding geocentric longitude.
    lon2 : array-like
        The other bounding gecentric longitude.
    shape : string (optional)
        'sphere' (default) or 'wgs84'
        'sphere' will assume a sphere with a radius of 6371008.7714 metres.
        'wgs84' will assume an oblate Earth with a semi-major axis 
        of 6378137.0 metres and a first eccentricity of 0.0818191908426215.
    angles : string (optional)
        'rad' (default) or 'deg'. 
        Specify if lat1, lat2, lon1 and lon2 are in degrees or radians.

    Returns
    -------
    aq : ndarray
        The area of the bounding box, given in square metres. Same shape
        as lat1, lat2, lon1 and lon2.

    Bryan Lougheed, February 2025

    This a python/numpy simplified port of the Octave function areaquad.Mi from the 
    Octave "mapping" package (v.1.4.2) (https://gnu-octave.github.io/packages/mapping/),
    which included the following license:

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    Full license text available at: http://www.gnu.org/licenses/
    """
    if angles == 'rad':
        lat1 = np.array(lat1)
        lat2 = np.array(lat2)
        lon1 = np.array(lon1)
        lon2 = np.array(lon2)
    elif angles == 'deg':
        lat1 = np.deg2rad(np.array(lat1))
        lat2 = np.deg2rad(np.array(lat1))
        lon1 = np.deg2rad(np.array(lat1))
        lon2 = np.deg2rad(np.array(lat1))
    else:
        raise Exception("'angles' parameter should be set to either 'deg' or 'rad'")

    if shape == 'sphere':
        a = 6371008.7714
        e = 0
    elif shape == 'wgs84':
        a = 6378137.0
        e = 0.0818191908426215
    else:
        raise Exception("'shape' parameter should be set to either 'sphere' or 'wgs84'")

    s1 = np.sin(lat1)
    s2 = np.sin(lat2)
    lonwidth = lon1 - lon2
    
    if e < np.finfo(float).eps:
        aq = abs((lonwidth * a**2) * (s2 - s1))
    else:
        e2 = e**2
        f = 1 / (2 * e)
        e2m1 = 1 - e2

        s21 = s1**2
        s22 = s2**2
        se1 = 1 - e2 * s21
        se2 = 1 - e2 * s22

        c = (lonwidth * a**2 * e2m1) / 2
        t1 = 1 + e * s1
        t2 = 1 + e * s2
        b1 = 1 - e * s1
        b2 = 1 - e * s2

        g = f * (np.log(t2 / b2) - np.log(t1 / b1))

        aq = np.abs( c * ((s2 / se2) - (s1 / se1) + g) )
  
    return aq


# def plotorbit(ecc, lpe, savename):
#     """
#     Creates a scale plot of the Earth's orbit and saves to hard drive.

#     Parameters
#     ----------
#     ecc : float
#         Eccentricity of the ellipse (ratio).
#     lpe : float
#         Longitude of perihelion as given by, e.g., Laskar: omega-bar 
#         (i.e. relative to NH autumn equinox) in radians.
#     savename : str
#         The name of the output .png file.
    
#     Output
#     ------
#     A scale plot of the orbit in current plotting window, or new plotting
#     window if there is none open.
#     """

#     # true anomaly v of seasons
#     omega = lpe + np.pi  # perihelion (geocentric)
#     nse = 2 * np.pi - omega  # nh spring equinox
#     nss = nse + np.pi/2  # nh summer solstice
#     nae = nss + np.pi/2  # nh autumn equinox
#     nws = nae + np.pi/2  # nh winter solstice

#     # convert true anomaly v to ellipse angle E (Meeus 1998 Equation 30.1 solve for E)
#     nse = 2 * np.arctan(  np.tan(nse / 2) * np.sqrt((1 - ecc)/(1 + ecc))  )
#     nss = 2 * np.arctan(  np.tan(nss / 2) * np.sqrt((1 - ecc)/(1 + ecc))  )
#     nae = 2 * np.arctan(  np.tan(nae / 2) * np.sqrt((1 - ecc)/(1 + ecc))  )
#     nws = 2 * np.arctan(  np.tan(nws / 2) * np.sqrt((1 - ecc)/(1 + ecc))  )

#     # semi-minor and semi-major axis of ellipse
#     a = 149.5978707  # au in 10^6 km
#     b = a*(1-ecc**2)**0.5
#     # perihelion and aphelion distances
#     rper = a * (1 - ecc)
#     raph = a * (1 + ecc)
#     fd = a - rper  # focal distance offset from centre of ellipse

#     # ellipse circumference points x and y coords
#     E = np.linspace(0, 2*np.pi, 1000)  # central angle E of ellipse, 0 to 360
#     xell = a*np.cos(E)-fd  # -fd to place sun at centre of image
#     yell = b*np.sin(E)

#     # plot
#     plt.figure()
#     plt.clf()

#     # plot the ellipse
#     plt.plot(xell, yell, '-', color=[0.6, 0.6, 0.6], linewidth=1)

#     # plot the sun
#     plt.plot(0, 0, 'yo', markersize=15, markerfacecolor=[255/255, 221/255, 66/255], markeredgecolor=[255/255, 143/255, 66/255])

#     # plot the astronomical seasons
#     ofs = 1.13  # text offset
    
#     def plot_season(v, label, color):
#         plt.plot(a * np.cos(v) - fd, b * np.sin(v), 'ko', markersize=7, markerfacecolor=color, markeredgecolor=[0, 100 / 255, 0])
#         plt.text(a * np.cos(v) * ofs - fd, b * np.sin(v) * ofs, label, horizontalalignment='center', rotation=np.rad2deg(v))

#     plot_season(nse, 'NH spring equinox', [0, 82 / 255, 162 / 255])
#     plot_season(nss, 'NH summer solstice', [0, 82 / 255, 162 / 255])
#     plot_season(nae, 'NH autumn equinox', [0, 82 / 255, 162 / 255])
#     plot_season(nws, 'NH winter solstice', [0, 82 / 255, 162 / 255])

#     # plot the per and aph distance
#     plt.arrow(xell[0], 0, -rper, 0, head_width=2, head_length=3, fc='r', ec='r')
#     plt.arrow(-fd, 0, raph, 0, head_width=2, head_length=3, fc='r', ec='r')
#     plt.text(np.mean([xell[0], 0]), 13, f'Perihelion\n{rper:.1f} million km', horizontalalignment='center')
#     plt.text(np.mean([a * np.cos(np.pi) - fd, 0]), 13, f'Aphelion\n{raph:.1f} million km', horizontalalignment='center')

#     plt.xlim([-170, 170])
#     plt.ylim([-170, 170])

#     plt.axis('off')
#     plt.gcf().set_size_inches(7.5, 7.5)
#     plt.savefig(savename, dpi=150, bbox_inches='tight')
#     plt.close()





# latres = 0.1
# totdays = 365.24
# dayres = 0.1
# lats = np.round(np.arange(90-latres/2,-90,-latres),2).reshape(-1,1)
# lats = np.round(np.arange(90-latres/2,-0,-latres),2).reshape(-1,1)
# gplats = np.rad2deg(geographiclat(lats,angles='deg'))

# # calculate area for each latband
# lataqo = areaquad(lats-latres/2, lats+latres/2, 0, 360, shape='wgs84', angles='deg')
# lataqs = areaquad(lats-latres/2, lats+latres/2, 0, 360, shape='sphere', angles='deg')


# # function for calculating Joblate and Jsphere for the latitude bands for certain sollon intervals
# def JoJs(lats, lataqs, lataqo, ecc, obl, lpe, totdays, dayres, con=1361, sollonmin=0, sollonmax=2*np.pi):
    
#     timerange = np.arange(0, totdays, dayres)
#     sollon, _ = time2sollon(time=timerange, ecc=ecc, lpe=lpe, tottime=totdays, obl=None, floatpp=64)
#     sollon = sollon[(sollon>=sollonmin) & (sollon<=sollonmax)].reshape(1,-1)
        
#     irr, _, _, _ = dailymeanwm2(lats, sollon, ecc, obl, lpe, con, earthshape='wgs84')
#     Qo = np.mean(irr,axis=1).reshape(-1,1)
#     Jo = np.mean(irr,axis=1).reshape(-1,1) * 24*60*60*dayres*irr.shape[1] * lataqo

#     irr, _, _, _ = dailymeanwm2(lats, sollon, ecc, obl, lpe, con, earthshape='sphere')
#     Qs = np.mean(irr,axis=1).reshape(-1,1)
#     Js = np.mean(irr,axis=1).reshape(-1,1) * 24*60*60*dayres*irr.shape[1] * lataqs
    
#     return Jo, Js, Qo, Qs


# # hemelrijck figure 3 check

# import matplotlib.pyplot as plt
# cm = 1/2.54

# ecc = 0.01672
# lpe = np.deg2rad(282.05-180)
# obl = np.deg2rad(23.45)

# figure2 = plt.figure(figsize=(10*cm, 21*cm))
# plt.clf()

# # annual
# _, _, Qo, Qs = JoJs(np.deg2rad(lats), lataqs, lataqo, ecc, obl, lpe, totdays, dayres, sollonmin=np.deg2rad(0), sollonmax=np.deg2rad(360))
# plt.plot(lats,100*(Qo-Qs)/Qs)
# #plt.plot(lats,Qo/Qs)
# #plt.plot(lats,Qo,'k-')
# #plt.plot(lats,Qs,'k--')
# # summer
# _, _, Qo, Qs = JoJs(np.deg2rad(lats), lataqs, lataqo, ecc, obl, lpe, totdays, dayres, sollonmin=np.deg2rad(0), sollonmax=np.deg2rad(180))
# plt.plot(lats,100*(Qo-Qs)/Qs)
# #plt.plot(lats,Qo/Qs)
# #plt.plot(lats,Qo,'r-')
# #plt.plot(lats,Qs,'r--')
# # winter
# _, _, Qo, Qs = JoJs(np.deg2rad(lats), lataqs, lataqo, ecc, obl, lpe, totdays, dayres, sollonmin=np.deg2rad(180), sollonmax=np.deg2rad(360))
# plt.plot(lats,100*(Qo-Qs)/Qs)
# #plt.plot(lats,Qo/Qs)
# #plt.plot(lats,Qo,'b-')
# #plt.plot(lats,Qs,'b--')

# #Qo, dayhrss, rxs, tsis = dailymeanwm2(np.deg2rad(lats), np.deg2rad(270), ecc, obl, lpe, earthshape='wgs84')
# #Qs, dayhrss, rxs, tsis = dailymeanwm2(np.deg2rad(lats), np.deg2rad(270), ecc, obl, lpe, earthshape='sphere')

# #plt.plot(lats,Qo)
# #plt.plot(lats,Qs)

# plt.grid(True)
# plt.xlim((0,90))
# #plt.ylim((-1.4,0.1))

# print(lats[(Qo-Qs)/Qs == np.min((Qo-Qs)/Qs)])


# ## plot matrix of dayints on x axis ant lats on y axis 
# timerange = np.arange(0, totdays, dayres)
# sollon, _ = time2sollon(time=timerange, ecc=ecc, lpe=lpe, tottime=totdays, obl=None, floatpp=64)
# sollon = sollon.reshape(1,-1)
# Qo, _, _, _ = dailymeanwm2(np.deg2rad(lats), sollon, ecc, obl, lpe, con=1361, earthshape='wgs84')
# Qs, _, _, _ = dailymeanwm2(np.deg2rad(lats), sollon, ecc, obl, lpe, con=1361, earthshape='sphere')

# diffmat = Qo/Qs
# #diffmat = 100*(Qo-Qs)/Qs
# diffmat[diffmat>1.0] = 999

# plt.figure()
# plt.imshow(diffmat, aspect='auto', cmap='viridis', extent=[np.min(timerange), np.max(timerange), np.min(lats), np.max(lats)])
# plt.colorbar()
# plt.show()
