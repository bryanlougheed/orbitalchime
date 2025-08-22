import numpy as np
import pandas as pd
from numba import njit, prange
from joblib import Parallel, delayed
import os
import warnings

def getlaskar2004(option=1, timeslice=(np.inf, -np.inf)):
    """
    tka, ecc, obl, lpe = getlaskar2004(option=1, timeslice=(np.inf, -np.inf))

    Open data files for the Laskar2004 solution (Laskar et al., 2004).
    Downloaded from http://vo.imcce.fr/insola/earth/online/earth/La2004/index.html

    Parameters
    ----------
    option : integer
        option = 1, 51 Ma to 0 Ma (default)
        option = 2, 0 Ma to 21 Ma in the future
        option = 3, 101 Ma to 0 Ma
        option = 4, 249 Ma to 0 Ma
        option = 5, 51 Ma to 21 Ma in the future (concatenate options 1 & 2)

    timeslice : array-like
        Contains one or two values (in ka before 2000 CE, negative values = future from 2000 CE)
        If one value, a single time interval. If two values, all time intervals spanning the two values.
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
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.

    Information
    -----------
    Script originally written in Matlab by Bryan Lougheed in 2020.
    Ported to python/pandas/numpy by Bryan Lougheed in Oct. 2024.

    Reference for the imported data:
    Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A.C.M., Levrard, B., 2004.
    "A long-term numerical solution for the insolation quantities of the Earth. "
    A&A 428, 261-285. https://doi.org/10.1051/0004-6361:20041335
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dirloc = file_dir+'/laskar_et_al/'

    def scinot(val): # because Laskar et al. output D (from Fortran) instead of E for overflow
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


def getlaskar1993(option=1, timeslice=(np.inf, -np.inf)):
    """
    tka, ecc, obl, lpe = getlaskar1993(option=1, timeslice=(np.inf, -np.inf))

    Open data files for the La93 solution (Laskar et al., 1993).
    Downloaded from http://vo.imcce.fr/insola/earth/online/earth/La2004/index.html

    Parameters
    ----------
    option : integer
        option = 1, La93(0,1) solution (default)
        option = 2, La93(1,1) solution

    timeslice : array-like
        Contains one or two values (in ka before 2000 CE, negative values = future from 2000 CE)
        If one value, a single time interval. If two values, all time intervals spanning the two values.
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
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.

    Information
    -----------
    Bryan Lougheed, July 2025.

    Reference for the imported data:
    Laskar, J., Joutel, F., Boudin, F., 1993. Orbital, precessional, and insolation quantities for the Earth 
    from -20 MYR to +10 MYR. Astronomy and Astrophysics 270, 522-533.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    dirloc = file_dir+'/laskar_et_al/'

    def scinot(val): # because Laskar et al output D (from fortran) instead of E for overflow
        return float(val.replace('D', 'E'))
 
    if option == 1:
        d1 = pd.read_csv(dirloc+'INSOLN.LA93_01.BTL.ASC', sep=r'\s+', header=None, skiprows=0, converters={1: scinot, 2: scinot, 3: scinot}).values
        d2 = pd.read_csv(dirloc+'INSOLP.LA93_01.BTL.ASC', sep=r'\s+', header=None, skiprows=0, converters={1: scinot, 2: scinot, 3: scinot}).values
        d = np.vstack((d1[1:], d2))
    elif option == 2:
        d1 = pd.read_csv(dirloc+'INSOLN.LA93_11.BTL.ASC', sep=r'\s+', header=None, skiprows=0, converters={1: scinot, 2: scinot, 3: scinot}).values
        d2 = pd.read_csv(dirloc+'INSOLP.LA93_11.BTL.ASC', sep=r'\s+', header=None, skiprows=0, converters={1: scinot, 2: scinot, 3: scinot}).values
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


def getlaskar2010(option=1, timeslice=(np.inf, -np.inf)):
    """
    tka, ecc = getlaskar2010(option=1, timeslice=(np.inf, -np.inf))

    Open Laskar et al. (2010) eccentricity solution data files. Useful for looking at eccentricity > 30 Ma.
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
    Script originally written in Matlab 2019a by Bryan Lougheed in 2020.
    Ported to python/pandas/numpy by Bryan Lougheed in Oct. 2024.
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

    # prep the data
    d[:, 0] = (d[:, 0] * -1) # geo style: make past positive, future negative
    d = d[d[:, 0].argsort()][::-1]  # sort rows by the first column in descending order

    # get the requested time slice
    timeslice = np.array([timeslice]) 
    d = d[(d[:, 0] >= np.min(timeslice)) & (d[:, 0] <= np.max(timeslice))]

    tka = np.array(d[:, 0])
    ecc = np.array(d[:, 1])

    return tka, ecc


def getBL1991(timeslice=(np.inf, -np.inf)):
    """
    tka, ecc = getBL1991(timeslice=(np.inf, -np.inf))

    Open data file for the Berger and Loutre (1991) astronomical solution.
    Downloaded from https://www.ncei.noaa.gov/pub/data/paleo/climate_forcing/orbital_variations/insolation/orbit91

    Parameters
    ----------
    timeslice : array-like
        Contains one or two values (in ka before 2000 CE, negative values = future from 2000 CE)
        If one value, a single time interval. If two values, all time intervals spanning the two values.
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
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.

    Information
    -----------
    Bryan Lougheed, Jul. 2025

    Reference for the imported data:
    Berger, A., Loutre, M.F., 1991. Insolation values for the climate of the last 
    10 million years. Quaternary Science Reviews 10, 297-317.
    https://doi.org/10.1016/0277-3791(91)90033-q

    """

    file_dir = os.path.dirname(os.path.abspath(__file__))
    dirloc = file_dir+'/berger_and_loutre/'
    
    d = pd.read_csv(dirloc+'orbit91.txt', sep=r'\s+', header=None, skiprows=4).values

    # prep the data
    d[:, 0] = (d[:, 0] * -1) # geo style: make past positive, future negative
    d = d[d[:, 0].argsort()][::-1]  # sort rows by the first column in descending order

    # get the requested time slice
    timeslice = np.array([timeslice]) 
    d = d[(d[:, 0] >= np.min(timeslice)) & (d[:, 0] <= np.max(timeslice))]

    # return np arrays
    tka = np.array(d[:, 0])
    ecc = np.array(d[:, 1])
    obl = np.deg2rad(np.array(d[:, 3]))
    lpe = np.deg2rad(np.array(d[:, 2]))

    return tka, ecc, obl, lpe


@njit(parallel=True)
def keplerjitloop(M, ecc, tolerance):
    # auxillary function called inside solvekeplerE(), the next function below

    # declare before linear loop
    n = M.size
    E = np.full(n, np.nan)
    precision = np.full(n, np.nan)
    pipi = 2*np.pi
    maxiters = maxiters = int((64 / 4) * 3.3 + 5)

    for i in prange(n):
    
        # get current M and ecc
        Mi = M[i]
        ecci = ecc[i]

        # Put in 0 to 2pi range
        signMi = 1.0
        if Mi < 0:
            Mi = -Mi
            signMi = -1.0
        Mi = (Mi % pipi) * signMi
        if Mi < 0:
            Mi += pipi

        # If inbound to perihelion, then we solve for the outbound symmetrical equivalent.
        # Will solve faster. Remember the sign (fE) to fix E at the end.
        fE = 1.0
        if Mi > np.pi:
            fE = -1.0
            Mi = pipi - Mi

        # First approximation, Lagrange expansion as detailed by, e.g., F.R. Moulton (1914) 
        # Moulton page 161, eq. 45
        Eo = Mi + ecci*np.sin(Mi) + 0.5*ecci**2*np.sin(2*Mi)  

        # Iterative "Differential corrections" detailed by Moulton using Taylor series
        for _ in np.arange(maxiters):
            M1 = Eo - ecci * np.sin(Eo) # Kepler equation
            dE = (Mi-M1) / (1-ecci*np.cos(Eo)) # Moulton page 162, eq. 47
            Eo += dE
            if np.abs(dE) <= tolerance: # when solved to within tolerance
                break                

        E[i] = Eo * fE
        precision[i] = dE

    return E, precision

def solvekeplerE(M, ecc, tolerance=1e-15, mode='auto'):
    """
    E, precision = solvekeplerE(M, ecc, tolerance=1e-15, mode='auto')

    Solves the Kepler equation for E to a very high precision.
    Tested as working for all possible Earth eccentricities. 
    
    Will probably work for higher eccentricities too (looks good up to 0.95), 
    but proceed at your own risk and check the function's 'precision' output.

    Parameters
    ----------
    M : ndarray
        Mean anomaly (radians)
    ecc : ndarray
        Eccentricity of the ellipse (ratio)
    tolerance : integer (optional)
        Minimum precision to converge on. Default is <1e-15.
        For reference, a 64-bit system has a precision of 2.2e-16.
        This option only works in 'numba' mode.
        'numpy' mode will always converge on <1e-15.
    mode : string (optional)
        'auto' (default), 'numba' or 'numpy'
        'auto' will automatically select 'numba' for large inputs

    Returns
    -------
    E : ndarray
        Eccentric anomaly (radians)
    precision : ndarray
        The value of the final differential correction on E (radians).

    Information
    -----------
    Python/numpy/numba implementaion of a Kepler equation solving procedure detailed by F.R. Moulton (1914).
    A Lagrange expansion is used to attain a first estimate of E, after which differential corrections are
    iteratively applied using a Taylor series until the desired precision for E is reached.

    References
    ----------
    F.R. Moulton (1914). "An Introduction to Celestial Mechanics", Second Revised Edition, The MacMillan Company, New York.
    Specifically: Section 95 (page 160) and Section 96 (page 162)
    """
    if mode == 'auto':
        if np.broadcast(M, ecc).size > 3*10**6:
            mode = 'numba'
        else:
            mode = 'numpy'

    if mode == 'numpy':
        
        # pure numpy fully vectorised version... runs on single thread :(
        M, ecc = np.broadcast_arrays(M, ecc)

        # put in 0 to 2pi range
        M = np.fmod(M, 2*np.pi)
        M[M < 0] += 2*np.pi
        
        # If inbound to perihelion, then we solve for the outbound symmetrical equivalent.
        # Will solve faster. Remember the sign (fE) to fix E at the end.
        fE = np.ones_like(M)
        mask = M>np.pi
        fE[mask] = -1
        M[mask] = 2*np.pi - M[mask]  
        
        # Informed first guess, Lagrange expansion as detailed by, e.g., F.R. Moulton (1914) 
        Eo = M + ecc*np.sin(M) + 0.5*ecc**2*np.sin(2*M) # Moulton page 161, eq. 45

        # Moulton "Differential corrections" using Taylor series
        for _ in np.arange(3): # this will get us to <1e-15
            M1 = Eo - ecc * np.sin(Eo) # Kepler equation
            dE = (M-M1) / (1-ecc*np.cos(Eo)) # Moulton page 162, eq. 47
            Eo += dE

        E = Eo * fE
        precision = dE

    elif mode == 'numba':

        # numpy+numba jit linear version optimised for multithreading

        # ensure inputs are arrays and broadcasted to same shape
        M, ecc = np.broadcast_arrays(np.asarray(M), np.asarray(ecc))
        M.setflags(write=True) # otherwise there will be a warning for some reason...
        ecc.setflags(write=True) # otherwise there will be a warning for some reason...
        M_flat = M.ravel() # Make 1D for numba
        ecc_flat = ecc.ravel() # Make 1D for numba

        # call keplerjitlooop numba function, which is defined
        # outside this function (above) so that it is compiled only once
        E_flat, prec_flat = keplerjitloop(M_flat, ecc_flat, tolerance)

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
        Keplerian geocentric solar longitude (lambda) in radians ('v' relative to NH spring equinox)
        Either 1 value (used as constant if other inputs are vector), or a vector of values.
    ecc : array-like
        Eccentricity (from, e.g., from Laskar et al.)
    lpe : ndarray
        heliocentric longitude of perihelion (from, e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
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
    
    The following sources were used (see also comments in script)
    
    For general texbook on Keplerian orbits:
    J. Meeus, (1998). Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia.
    Specifically Chapter 30.
    
    For calculating equation of time:
    P. Edwards: https://dr-phill-edwards.eu/Astrophysics/EOT.html
    """

    # in case lpe is only one value
    lpe = np.atleast_1d(lpe)   

    # Change lpe from heliocentric to geocentric
    lpegeo = lpe + np.pi
    lpegeo[lpegeo >= 2*np.pi] -= 2*np.pi  # wrap to 360

    # Get day of anchor day (dz) relative to perihelion
    vz = 2*np.pi - lpegeo  # v of spring equinox relative to perihelion
    vz[vz > 2*np.pi] -= 2*np.pi
    Ez = 2 * np.arctan(np.tan(vz/2) * np.sqrt((1-ecc) / (1+ecc)))  # Meeus (1998) page 195, solve for E
    Mz = Ez - ecc * np.sin(Ez)  # Meeus page 195, solve for M (Kepler equation). M is the circular orbit equivalent of v
    Mz[Mz < 0] = np.pi + (np.pi - Mz[Mz<0] * -1)  # inbound to perihelion
    dz = Mz / (2*np.pi) * tottime

    # Get day of target day (dx) relative to perihelion
    vx = vz + sollon
    vx[vx>2*np.pi] -= 2 * np.pi
    Ex = 2 * np.arctan(np.tan(vx / 2) * np.sqrt((1-ecc) / (1+ecc)))  # Meeus (1998) page 195, solve for E
    Mx = Ex - ecc * np.sin(Ex)  # Solve Kepler equation for M
    Mx[Mx<0] = np.pi + (np.pi - Mx[Mx<0] * -1)  # inbound to perihelion, (probably not necessary)
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
        dtecc = np.rad2deg(Mx-vx) * 4  # four minutes per degree longitude (24 hrs * 60 mins / 360 degrees )
        # obliquity component
        alpha = np.arctan2(np.sin(sollon) * np.cos(obl), np.cos(sollon))
        alpha[alpha<0] += 2*np.pi
        dtobl = np.rad2deg(sollon-alpha) * 4 # same here
        # total EOT, time in minutes
        eot = dtecc + dtobl
    else:
        eot = np.array([])

    return time, eot

def time2sollon(time, ecc, lpe, tottime=365.24, obl=None):
    """
    sollon, eot = time2sollon(time, ecc, lpe, tottime=365.24, obl=None)

    Given a particular eccentricity and longitude of perihelion, get geocentric solar longitude 
    associated with a particular time of the tropical year i.e. by accounting for 
    conservation of angular momentum during orbit (Kepler's 2nd Law).

    Parameters
    ----------
    time : ndarray
        Time interval of tropical year (where interval 0 is boreal spring equinox). Either on value, or vector of values.
    ecc : ndarray
        Eccentricity (e.g., from Laskar et al.)
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    tottime : float
        Total time in the year corresponding to 'time', single value, any time unit you want. Default value is 365.24.
    obl : ndarray, optional
        Obliquity in radians (e.g., from Laskar et al.) for calculating the equation of time (eot)

    Input can be vectorised in various ways, but double check output.

    Returns
    -------
    sollon : ndarray
        Keplerian geocentric solar longitude in radians ('lambda', i.e. 'v' relative to boreal spring equinox) 
    eot : ndarray
        Equation of time (minutes). Returns empty if obl not supplied.

    Info
    ----
    Bryan Lougheed, June 2020, Matlab 2019a
    Updated April 2023 to include eot.
    Ported to python/numpy in October 2024.
    Updated 2025 with more efficient Kepler equation solution.
    Python 3.12.4, numpy 1.26.4.

    The following sources were used (see also comments in script)
    
    For general texbook on Keplerian orbits:
    J. Meeus, (1998). Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia.
    Specifically: Chapter 30.
    
    For Kepler equation solution:
    F.R. Moulton, (1914). "An Introduction to Celestial Mechanics", 2nd. rev. ed., The MacMillan Company, New York.
    Specifically: Section 95 (page 160) and Section 96 (page 162)
    
    For calculating equation of time:
    P. Edwards: https://dr-phill-edwards.eu/Astrophysics/EOT.html
    """
    # convert input to numpy for speed and compatibility
    tottime = np.array([tottime])
    time = np.atleast_1d(time).reshape(-1,1)

    # in case lpe is only one value
    lpe = np.atleast_1d(lpe)

    # change lpe from heliocentric to geocentric
    lpegeo = lpe + np.pi # np.array needed for when doing only one timeslice
    lpegeo[lpegeo >= 2*np.pi] = lpegeo[lpegeo >= 2*np.pi] - 2*np.pi

    # NH spring equinox relative to perihelion
    veq = 2*np.pi - lpegeo
    Eeq = 2 * np.arctan(np.tan(veq/2) * np.sqrt((1-ecc) / (1+ecc)))
    Meq = Eeq - ecc * np.sin(Eeq) # as previous comment
    Meq[Meq<0] = np.pi + (np.pi - Meq[Meq<0] * -1)
    deq = Meq / (2*np.pi) * tottime

    # v of target (x) v relative to perihelion
    deq, time = np.broadcast_arrays(deq,time)
    dx = deq + time
    Mx = (dx / tottime) * 2*np.pi
    Ex, _ = solvekeplerE(Mx, ecc, tolerance=1e-15)  # Solve Kepler equation for E
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
    gplat = np.arctan((re/rp)**2 * np.tan(gclat))

    return gplat

def dailymeanwm2(lat, sollon, ecc, obl, lpe, con=1361, earthshape='sphere'):
    """
    dmwm2, dayhrs, rx, tsi = dailymeanwm2(lat, sollon, ecc, obl, lpe, con=1361, earthshape='sphere')

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
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    con : float or array-like, optional
        Solar constant in W/m². Single numerical value or 1D array. Default is 1361 W/m².
    earthshape : str, optional
        Shape of Earth, either 'sphere' or 'wgs84' for wgs84 ellipsoid (default is 'sphere').

    Returns
    -------
    dmwm2, dayhrs, tsi, rx

    dmwm2 : ndarray
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
    Bryan Lougheed, May 2020, Matlab 2019a
    Updated to include Earth's oblateness Sep. 2020
    Ported to python/numpy Oct. 2024 by Bryan Lougheed
    Python 3.12.4, numpy 1.26.4.

    Daily mean irradiance for a spherical Earth calculated following Milanković (1941).
    See Part 4 therein ("Vierter Abschnitt: Die Bestrahlung der Erde durch die Sonne und die säkularen
    enderungen dieser Bestrahlung"), specifically how equation 22 is derived.
    
    Option to calculate daily mean irradiance in the case of the oblate Earth addded 
    following a procedure by Van Hemelrijck (1983).
    
    Standard Keplerian orbital mechanics detailed in, e.g., Meeus (1998) were also used.

    References
    ----------
    Milanković, M., 1941. Kanon der Erdebestrahlung und seine Anwendung auf das Eiszeitenproblem. Königlich Serbische Akademie.
    Van Hemelrijck, E., 1983. The oblateness effect on the extraterrestrial solar radiation. Solar Energy 31, 223-228. https://doi.org/10.1016/0038-092X(83)90085-3
    Meeus, J., 1998. Astronomical Algorithms, 2nd ed. Willmann-Bell, Inc., Richmond, Virginia.

    Daylight hours output following sunrise equation: https://en.wikipedia.org/wiki/Sunrise_equation
     
    """
    # numpy everything just to be sure
    lat = np.array(lat)
    sollon = np.array(sollon)
    ecc = np.array(ecc)
    obl = np.array(obl)
    lpe = np.array(lpe)
    lpe = np.atleast_1d(lpe)
    con = np.array(con)
    
    # Calculate rx and tsi using Keplerian orbital mechanics detailed by e.g. Meeus (1998)
    lpegeo = lpe + np.pi  # add 180 degrees. (heliocentric to geocentric)
    lpegeo[lpegeo >= 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    veq = 2*np.pi - lpegeo  # v (true anomaly) of spring equinox relative to perihelion
    vx = veq + sollon  # v (true anomaly) of inputted sollon relative to perihelion
    vx[vx > 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    rx = (1 - ecc**2) / (1 + ecc * np.cos(vx))  # Distance from Sun in AU, Eq. 30.3 in Meeus (1998)
    tsi = con * (1 / rx)**2 # Total solar irradiance at distance rx

    # for calculating daylight hours
    dlen = 24 # placeholder for future upgrade, only 24 hour days have been tested so far

    if earthshape == 'sphere':

        # Declination angle of the sun
        # https://en.wikipedia.org/wiki/Position_of_the_Sun
        dsun = np.arcsin(np.sin(obl) * np.sin(sollon))

        # Hour angle at sunrise/sunset
        # https://en.wikipedia.org/wiki/Sunrise_equation
        # See also Milanković (1941), Part 4 (Vierter Abschnitt), eq. 20
        # numpy will return NaN for invalid input to arccos caused by polar day or polar night, we will use it to our advantage later.
        # numpy NaN output warning supressed below
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hangle = np.arccos(-np.tan(lat) * np.tan(dsun))
        
        # replace NaNs in hangle from above where it is polar day or polar night
        # If lat*dsun is >0 then polar region and an subsolar point must both be in same hemisphere --> polar day --> pi
        # otherwise lat*dsun <0 --> polar night --> 0
        # https://en.wikipedia.org/wiki/Hour_angle
        hangle[np.logical_and(np.isnan(hangle), lat*dsun > 0)] = np.pi # polar day
        hangle[np.logical_and(np.isnan(hangle), lat*dsun <= 0)] = 0    # polar night
        # Where the user intentionally suppplied NaN for lat, obl or sollon, the above won't trigger (because lat*dsun will be nan), meaning the user's NaN placement is preserved
        
        # Hours of daylight (https://en.wikipedia.org/wiki/Sunrise_equation)
        dayhrs = np.abs(hangle - hangle*-1) / (2*np.pi / dlen)

        # Daily mean irradiance
        # "mittlere Bestrahlung" in Milanković (1941)
        # Equation is derived in Part 4 (Vierter Abschnitt), leading to eq. 22 therein
        # Below, mean irradiation at distance rx (Jo/rx^2) is substituted for mean irradiance at distance rx, i.e. tsi calculated earlier
        dmwm2 = (1/np.pi) * tsi * ( hangle * np.sin(lat) * np.sin(dsun) + np.cos(lat) * np.cos(dsun) * np.sin(hangle) )

    elif earthshape == 'wgs84':
     
        # Van Hemelrijck (1983) extended method for oblate Earth

        # Declination angle of the sun
        # https://en.wikipedia.org/wiki/Position_of_the_Sun
        dsun = np.arcsin(np.sin(obl) * np.sin(sollon))
        # Hour angle at sunrise/sunset
        # https://en.wikipedia.org/wiki/Sunrise_equation
        # use geogrpaphic lat following Van Hemelrijck
        gglat = geographiclat(lat)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hangle = np.arccos(-np.tan(gglat) * np.tan(dsun))

        # https://en.wikipedia.org/wiki/Hour_angle
        hangle[np.logical_and(np.isnan(hangle), lat*dsun > 0)] = np.pi  # polar day
        hangle[np.logical_and(np.isnan(hangle), lat*dsun <= 0)] = 0    # polar night
        
        # Hours of daylight (https://en.wikipedia.org/wiki/Sunrise_equation)
        dayhrs = np.abs(hangle - hangle*-1) / (2*np.pi / dlen)
      
        f = 1 / 298.257223563 # wgs84 flattening value
        vangle = np.arctan((1 - f)**-2 * np.tan(lat)) - lat  # Van Hemelrijck (1983) eq. 9, f is wgs84 flattening
        dmwm2 = (1/np.pi) * tsi * (np.cos(vangle) * (hangle * np.sin(lat) * np.sin(dsun) + np.sin(hangle) * np.cos(lat) * np.cos(dsun)) + np.sin(vangle) * (-np.tan(lat) * (hangle * np.sin(lat) * np.sin(dsun) + np.sin(hangle) * np.cos(lat) * np.cos(dsun)) + hangle * np.sin(dsun) / np.cos(lat)))
           
    else:
        raise ValueError('earthshape '+earthshape+' unrecognised, should be sphere or wgs84')


    return dmwm2, dayhrs, rx, tsi

def intradaywm2(lat, ecc, obl, lpe, dayint, daysinyear=365.24219, con=1361.0):
    """
    idwm2, elev, msdhr, lashr, eot = intradaywm2(lat, ecc, obl, lpe, dayint, daysinyear=365.242, con=1361.0)

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
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    dayint : ndarray
        The mean solar day interval(s) to be analysed, in day decimals (i.e. 0.5 for midday on day 0).
        Note that, due to the equation of time, 0.5 would not necessarily exactly correspond to local solar noon.
        Calculations will assume that dayint 0.0 (midnight on day zero) corresponds to the northern hemisphere
        spring equinox.
    daysinyear : float
        The number of mean solar days in the year, default is 365.24219
    con : float
        solar constant in W/m², default is 1361 W/m²

    Returns
    -------
    idwm2, elev, msdhr, lashr, eot

    idwm2 : ndarray
        array of W/m² for every mean solar day interval calculated
    elev : ndarray
        solar elevation (radians), same dimension as idwm2
    msdhr : ndarray
        time of the the mean solar day (0-24, in hours), same dimension as idwm2
        Can be best thought of as "clock hours".
    lashr : ndarray
        local apparent solar hour (0-24), same dimension as idwm2
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
    lpegeo = lpe + np.pi
    veq = 2*np.pi - lpegeo  # v (true anomaly) of NH spring equinox relative to perihelion
    vx = veq + sunlon  # v (true anomaly) of inputted sunlon relative to perihelion
    vx[vx > 2*np.pi] -= 2*np.pi  # put back in 0-360 range
    rx = (1-ecc**2) / (1 + ecc*np.cos(vx))  # Eq. 30.3 in Meeus (1998)

    # Calculate tsi as function of con relative to 1 AU
    tsi = con * (1/rx)**2

    # Calculate TOA W/m2, i.e. the vertical component of TSI W/m2
    idwm2 = tsi * np.sin(elev)
    idwm2[idwm2 < 0] = 0  # sun under horizon, night time

    return idwm2, elev, msdhr, lashr, eot


def daytoday(lat, day1, day2, ecc, obl, lpe, dayres=0.1, totdays=365.2, con=1361, earthshape='sphere'):
    """
    daysjm2, lambda1, lambda2 = daytoday(lat, day1, day2, ecc, obl, lpe, dayres=0.1, totdays=365.2, con=1361, earthshape='sphere')

    Calculate integrated irradiation (J/m²) at top of atmosphere between two mean solar day values of the tropical year.
    
    Parameters
    ----------
    lat : float
        Geocentric latitude (in radians, N is positive, negative for S) on Earth. Single value.
    day1 : float
        Mean solar day (or fraction thereof, e.g. 102.52) from which to start the integration. Single value.
        Mean solar day 0.0 is assumed to occur at the NH spring equinox.
    day2 : float
        Mean solar day (or fraction thereof, e.g. 202.52) from which to start the integration. Single value.
        If day2 < day1, the function will assume that you wish to integrate across tropical day 0.
    ecc : ndarray
        Eccentricity. Numerical value(s). 1D array.
    obl : ndarray
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    dayres : float, optional
        Mean solar day resolution for the integration. Default is 0.1.
    totdays : float, optional
        Total mean solar days in the tropical year. Default is 365.2.
    con : float or array-like, optional
        Solar constant. Single numerical value or 1D array, W/m2. Default is 1361.
    earthshape : str (optional)
        Shape of Earth, 'sphere' (default) or 'wgs84'.

    Returns
    -------
    daysjm2 : ndarray
        Integrated irradiation (J/m2) at top of atmosphere between day1 and day2.
    lambda1 : ndarray
        Solar longitude (λ, in radians) corresponding to day1.
    lambda2 : ndarray
        Solar longitude (λ, in radians) corresponding to day2.
    """

    if day1 > totdays or day2 > totdays:
        raise Exception("day1 or day2 cannot be greater than value totdays")
    elif day1 < 0 or day2 < 0:
        raise Exception("day1 or day2 cannot be less than zero")

    lambda1, _ = time2sollon(day1, ecc, lpe, tottime=totdays)
    lambda2, _ = time2sollon(day2, ecc, lpe, tottime=totdays)

    if day2 >= day1:

        sollons, _ = time2sollon(np.arange(day1,day2+dayres,dayres), ecc, lpe, tottime=totdays)
        dmwm2, _, _, _ = dailymeanwm2(lat=lat, sollon=sollons, ecc=ecc, obl=obl, lpe=lpe, con=con, earthshape=earthshape)
        daysjm2 = np.mean(dmwm2, axis=0) * (np.abs(day2-day1)/totdays)*(365.24219*86400)
        # W/m2 * the fraction of the year integrated by by day2-day1 * the total SI seconds in the year = J/m2
        # the number of SI seconds in a year should in principle be constant at 365.24219*86400 for other day lengths also
        # unless e.g. semi-major axis changes
    
    elif day2 < day1:

        # Part A: day1 to end of tropical year
        sollons, _ = time2sollon(np.arange(day1,totdays,dayres), ecc, lpe, tottime=totdays)
        dmwm2, _, _, _ = dailymeanwm2(lat=lat, sollon=sollons, ecc=ecc, obl=obl, lpe=lpe, con=con, earthshape=earthshape)
        daysjm2A = np.mean(dmwm2, axis=0) * (np.abs(totdays-dayres-day1)/totdays)*(365.24219*86400)

        # Part B: start of tropical year to day2
        sollons, _ = time2sollon(np.arange(0,day2+dayres,dayres), ecc, lpe, tottime=totdays)
        dmwm2, _, _, _ = dailymeanwm2(lat=lat, sollon=sollons, ecc=ecc, obl=obl, lpe=lpe, con=con, earthshape=earthshape)
        daysjm2B = np.mean(dmwm2, axis=0) * (np.abs(day2-0)/totdays)*(365.24219*86400)

        daysjm2 = daysjm2A + daysjm2B

    return daysjm2, lambda1, lambda2


def lambdatolambda(lat, lambda1, lambda2, ecc, obl, lpe, dayres=0.1, totdays=365.2, con=1361, earthshape='sphere'):
    """
    lambdajm2, day1, day2 = lambdatolambda(lat, lambda1, lambda2, ecc, obl, lpe, dayres=0.1, totdays=365.2, con=1361, earthshape='sphere'):

    Calculate integrated irradiation (J/m²) at top of atmosphere between two given solar longitudes (λ).
    
    Parameters
    ----------
    lat : float
        Geocentric latitude (in radians, N is positive, negative for S) on Earth. Single value.
    lambda1 : float
        Solar longitude (λ, in radians) from which to start the integration. Single value.
    lambda2 : float
        Solar longitude (λ, in radians) at which to finish the integration. Single value.
        If lambda2 < lambda1, the function will assume that you wish to integrate across tropical day 0.
    ecc : ndarray
        Eccentricity. Numerical value(s). 1D array.
    obl : ndarray
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    dayres : float, optional
        Mean solar day resolution for the integration. Default is 0.1.
    totdays : float, optional
        Total mean solar days in the tropical year. Default is 365.2.
    con : float or array-like, optional
        Solar constant. Single numerical value or 1D array, W/m2. Default is 1361.
    earthshape : str (optional)
        Shape of Earth, 'sphere' (default) or 'wgs84'.

    Returns
    -------
    lambdajm2 : ndarray
        Integrated irradiation (J/m2) at top of atmosphere between lambda1 and lambda2
    day1 : ndarray
        Mean solar day interval corresponding to lambda1
    day2 : ndarray
        Mean solar day interval corresponding to lambda2
    """

    day1, _ = sollon2time(lambda1, ecc, lpe, tottime=totdays)
    day2, _ = sollon2time(lambda2, ecc, lpe, tottime=totdays)

    mode = 'parallel'

    if mode == 'serial':
        lambdajm2 = np.full(ecc.shape, np.nan)
        for i in np.arange(ecc.size):
            lambdajm2[i], _, _ = daytoday(lat, day1[i], day2[i], ecc[i], obl[i], lpe[i], dayres=dayres, totdays=totdays, con=con, earthshape=earthshape)

    elif mode == 'parallel':
        def par_lambdajm2(i, lat, day1, day2, dayres, ecc, obl, lpe, totdays, con, earthshape):
            result, _, _ = daytoday(lat, day1[i], day2[i], ecc[i], obl[i], lpe[i], dayres=dayres, totdays=totdays, con=con, earthshape=earthshape)
            return result

        results = Parallel(n_jobs=-1)(
            delayed(par_lambdajm2)(
                i, lat, day1, day2, dayres, ecc, obl, lpe, totdays, con, earthshape
            ) for i in range(ecc.size)
        )

        lambdajm2 = np.array(results).squeeze()

    return lambdajm2, day1, day2


def thresholdjm2(thresh, lat, ecc, obl, lpe, timeres=0.1, tottime=365.2, con=1361, earthshape='sphere'):
    """
    threshjm2, timethresh = thresholdjm2(thresh, lat, ecc, obl, lpe, timeres=0.1, tottime=365.24, con=1361, earthshape='sphere')

    Calculate integrated irradiation (J/m²) at top of atmosphere for all day intervals 
    exceeding a certain threshold in mean daily irradiance (W/m²).
    Can be used to emulate analysis by, e.g., Huybers (2006; 10.1126/science.1125249)
    
    Parameters
    ----------
    thresh : float or array-like
        Threshold value (W/m2). Single value, or vector of values.
    lat : float
        Geocentric latitude (in radians, N is positive, negative for S) on Earth. Single value.
    ecc : array-like
        Eccentricity. Numerical value(s). 1D array.
    obl : array-like
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    timeres : float, optional
        Time resolution for the integration. Default is 0.1.
    tottime : float, optional
        Total time in the tropical year. Default is 365.2.
    con : float or array-like, optional
        Solar constant. Single numerical value or 1D array, W/m2. Default is 1361.
    earthshape : str (optional)
        Shape of Earth, 'sphere' (default) or 'wgs84'.

    Returns
    -------
    threshjm2 : ndarray
        Integrated irradiation at top of atmosphere for days exceeding thresh. J/m2. Array same dimensions as ecc, obl, and lpe.
    timethresh : ndarray
        Time exceeding thresh (in seconds, assuming 365.24219*86400 seconds in a year).
        Same dimensions as threshjm2.
    """

    timerange = np.arange(0, tottime, timeres)
    sollons, _ = time2sollon(timerange, ecc, lpe, tottime)
    irrs, _, _, _ = dailymeanwm2(lat, sollons, ecc, obl, lpe, con=con, earthshape=earthshape)
    timethresh = (np.sum(irrs > thresh, axis=0) / timerange.size) * 365.24219*86400
    threshirrs = np.where(irrs > thresh, irrs, np.nan)
    threshjm2 = np.nanmean(threshirrs, axis=0) * timethresh  # W/m2 to J/m2
    
    return threshjm2, timethresh


@njit(parallel=True)
def jitloopshj(irrs):
    nints, nkyr = irrs.shape
    halfnints = nints // 2
    halfyrsec = (365.24219*86400) / 2
    shjjm2 = np.full(nkyr, np.nan)

    for i in prange(nkyr):
        col = irrs[:, i]
        for j in range(1, nints - halfnints):
            summer = col[j:j + halfnints]
            winpre = col[:j]
            winpost = col[j + halfnints:]
            winmax = max(np.max(winpre), np.max(winpost))

            if np.min(summer) > winmax:
                shjjm2[i] = np.mean(summer) * halfyrsec # W/m2 to J/m2
                break
    return shjjm2

def sommerhalbjahr(lat, ecc, obl, lpe, timeres=0.1, tottime=365.24, con=1361, earthshape='sphere'):
    """
    shjjm2 = sommerhalbjahr(lat, ecc, obl, lpe, timeres=0.1, tottime=365.24, con=1361, earthshape='sphere')

    Only works at northern high latitudes.
    
    Parameters
    ----------
    lat : float
        Geocentric latitude (in radians, N is positive, negative for S) on Earth. Single value.
    ecc : array-like
        Eccentricity. Numerical value(s). 1D array.
    obl : array-like
        Obliquity. Numerical value(s), radians. 1D array.
    lpe : ndarray
        heliocentric longitude of perihelion (from e.g., Laskar et al.)
        ϖ (i.e., 'v' relative to NH autumn equinox) in radians.
    con : float or array-like, optional
        Solar constant. Single numerical value or 1D array, W/m². Default is 1361.
    timeres : float, optional
        Day resolution for the integration. Default is 0.1.
    tottime : float, optional
        Total time in the tropical year. Default is 365.24.
    earthshape : str (optional)
        Shape of Earth, 'sphere' (default) or 'wgs84'.

    Returns
    -------
    shjjm2 : ndarray
        Milanković caloric summer half year (sommerhalbjahr) in J/m². Array same dimensions as ecc, obl, and lpe.
    """

    timeints = np.arange(0,tottime,timeres)
    sollons, _ = time2sollon(timeints, ecc, lpe, tottime=tottime)
    irrs, _, _, _ = dailymeanwm2(lat=lat, sollon=sollons, ecc=ecc, obl=obl, lpe=lpe, con=con, earthshape=earthshape)
    ind = timeints > (270/365)*tottime
    irrs = np.vstack((irrs[ind, :], irrs[~ind, :]))
    shjjm2 = jitloopshj(irrs)

    return shjjm2


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

    This a python/numpy simplified port of the Octave function areaquad.m from the 
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
        aq = abs((lonwidth * a**2) * (s2-s1))
    else:
        e2 = e**2
        f = 1 / (2*e)
        e2m1 = 1-e2

        s21 = s1**2
        s22 = s2**2
        se1 = 1-e2*s21
        se2 = 1-e2*s22

        c = (lonwidth * a**2 * e2m1) / 2
        t1 = 1+e*s1
        t2 = 1+e*s2
        b1 = 1-e*s1
        b2 = 1-e*s2

        g = f * (np.log(t2/b2) - np.log(t1/b1))

        aq = np.abs( c * ((s2/se2) - (s1/se1) + g) )
  
    return aq

