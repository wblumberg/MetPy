# Copyright (c) 2008-2015 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import division

import numpy as np

from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist

from metpy.gridding import interpolation
from metpy.gridding import points

import bisect

from ..package_tools import Exporter

import matplotlib.tri as tri

exporter = Exporter(globals())

def calc_avg_spacing(x, y, filter=False):
    r"""Calculate the average spacing between the stations using nearest neighbor and Delaunay Triangulation

    Parameters
    ----------
    x: ndarray
        a list of the x points for the stations
    y: ndarray
        a list of the y points for the stations
    filter: bool
        filter out any outliers from the triangulation
    
    Returns
    -------
    avg_spacing: float
        the average spacing (nearest neighbor)
    """
    
    triang = tri.Triangulation(x,y)
    for edge in triang.edges:
        edge_length.append(np.sqrt(np.power(x[edge[0]] - x[edge[1]],2) + np.power(y[edge[0]] - y[edge[1]], 2)))
    if filter is True:
        per = np.percentile(edge_length,(10,90))
        i = np.where((edge_length > per[0]) & (edge_length < per[1]))[0]
    else:
        i = np.arange(len(edge_length))

    avg_spacing = np.mean(edge_length[i])

    return avg_spacing

def estimate_grid_dx(x, y, avg_spacing=None):
    r"""Calculate the recommended grid spacing using the 
        Koch et al. 1982 recommendations.
        
        Parameters
        ----------
        x: ndarray
            a list of the x points for the stations
        y: ndarray
            a list of the y points for the stations
        avg_spacing: float
            the average spacing between stations
                
        Returns
        -------
        None
    """
    if avg_spacing is None:
        avg_spacing = calc_avg_spacing(x, y)
    #avg_spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y))))) 
    lower_bound = avg_spacing * (1./3.)
    upper_bound = avg_spacing * 0.5
    print "Recommended grid spacing is between:", round(lower_bound, 2), "and", round(upper_bound), "km"

def calc_kappa(spacing, kappa_star=5.052):
    r"""Calculate the kappa parameter for barnes interpolation.

    From Koch et al. 1982, JAMC (Eq. 13)
    
    Parameters
    ----------
    spacing: float
        Average spacing between observations
    kappa_star: float
        Non-dimensional response parameter. Default 5.052.

    Returns
    -------
        kappa: float
    """

    return kappa_star * (2.0 * spacing / np.pi)**2


def remove_observations_below_value(x, y, z, val=0):
    r"""Given (x,y) coordinates and an associated observation (z),
    remove all x, y, and z where z is less than val. Will not destroy
    original values.

    Parameters
    ----------
    x: float
        x coordinate.
    y: float
        y coordinate.
    z: float
        Observation value.
    val: float
        Value at which to threshold z.

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        observation values less than val.
    """

    x_ = x[z >= val]
    y_ = y[z >= val]
    z_ = z[z >= val]

    return x_, y_, z_


def remove_nan_observations(x, y, z):
    r"""Given (x,y) coordinates and an associated observation (z),
    remove all x, y, and z where z is nan. Will not destroy
    original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        nan valued observations.
    """

    x_ = x[~np.isnan(z)]
    y_ = y[~np.isnan(z)]
    z_ = z[~np.isnan(z)]

    return x_, y_, z_


def remove_repeat_coordinates(x, y, z):
    r"""Given x,y coordinates and an associated observation (z),
    remove all x, y, and z where (x,y) is repeated and keep the
    first occurrence only. Will not destroy original values.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value

    Returns
    -------
    x, y, z
        List of coordinate observation pairs without
        repeated coordinates.
    """

    coords = []
    variable = []

    for (x_, y_, t_) in zip(x, y, z):
        if (x_, y_) not in coords:
            coords.append((x_, y_))
            variable.append(t_)

    coords = np.array(coords)

    x_ = coords[:, 0]
    y_ = coords[:, 1]

    z_ = np.array(variable)

    return x_, y_, z_

def define_grid(x,y,hres,buffer,lbound,rbound,tbound,bbound):
    r"""Define the grid for an analysis.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    hres: float
        The horizontal resolution of the generated grid. Default 50000 meters.
    buffer: float
        How many meters to add to the bounds of the grid. Default 1000 meters.
    lbound : float
        number of meters to trim off of left side of box
    rbound : float
        number of meters to trim off of right side of box
    tbound : float
        number of meters to trim off the top part of the box
    bbound : float
        number of meters to trim off of the bottom part of the box

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, N) ndarray
        2-dimensional array representing the interpolated values for each grid.
    """


    grid_x, grid_y = points.generate_grid(hres, points.get_boundary_coords(x, y),
                                             buffer)
    grid_x, grid_y = points.trim_grid(grid_x, grid_y, lbound, rbound, tbound, bbound)
    
    return (grid_x, grid_y)

def bilinear_interp(x, y, x_grid, y_grid, z):
    if x > np.max(x_grid[0,:]) or x < np.min(x_grid[0,:]) or y > np.max(y_grid[:,0]) or y < np.min(y_grid[:,0]):
        #print x, np.max(x_grid[0,:]), x, np.min(x_grid[0,:]), y, np.max(y_grid[:,0]), y, np.min(y_grid[:,0])
        #print "Excluding:", x, y
        # return masked if the point is outside the grid.
        return np.ma.masked
    x_mid = np.interp(x, x_grid[0,:], z[0,:])
    y_mid = np.interp(y, y_grid[:,0], z[:,0])

    x_right = bisect.bisect_left(x_grid[0,:], x)
    x_left = bisect.bisect_right(x_grid[0,:], x)-1
    y_top = bisect.bisect_left(y_grid[:,0], y)
    y_bottom = bisect.bisect_right(y_grid[:,0], y)-1

    gridpt_11 = (x_grid[y_top, x_right], y_grid[y_top, x_right], z[y_top, x_right])
    gridpt_21 = (x_grid[y_top, x_left], y_grid[y_top, x_left], z[y_top, x_left])
    gridpt_12 = (x_grid[y_bottom, x_right], y_grid[y_bottom, x_right], z[y_bottom, x_right])
    gridpt_22 = (x_grid[y_bottom, x_left], y_grid[y_bottom, x_left], z[y_bottom, x_left])
    return (gridpt_11[2] * (gridpt_21[0] - x) * (gridpt_12[1] - y) +
            gridpt_21[2] * (x - gridpt_11[0]) * (gridpt_12[1] - y) +
            gridpt_12[2] * (gridpt_21[0] - x) * (y - gridpt_11[1]) +
            gridpt_22[2] * (x - gridpt_11[0]) * (y - gridpt_11[1])
           ) / ((gridpt_21[0] - gridpt_11[0]) * (gridpt_12[1] - gridpt_11[1]) + 0.0)

def multi_pass_barnes(x, y, z, grid, interp_type, passes=2, min_neighbors=3, guess=None, gamma=0.5, kappa_star=5.052, ave_spacing=None, search_radius=None):
    r"""
    """
    if guess is None:
        guess = np.zeros(grid[0].shape)
    g = gamma
    for i in xrange(passes):
        if i == 0:
            gamma = 1
            obs = z
            valid_obs = np.arange(len(obs))
        else:
            gamma = g
            obs = z - back_interp
            valid_obs = ~back_interp.mask
        #print valid_obs
        #print len(valid_obs)
        #print obs[valid_obs]
        # Solve for the grid ( do the analysis ) and update guess.
        result = interpolate(x[valid_obs], y[valid_obs], obs[valid_obs], guess, grid, interp_type='barnes', ave_spacing=ave_spacing, search_radius=search_radius, gamma=gamma)[2]
        #print result
        # Back interpolate to the grid points.
        analysis = []
        for x_o, y_o, z_o in zip(x,y,z):
            analysis.append(bilinear_interp(x_o, y_o, grid[0], grid[1], result))
        back_interp = np.ma.asarray(analysis)
        print "Pass:", i, "RMS:", round(np.ma.std(z - back_interp),3)
        # Update the first guess
        guess = result
    return result
        

@exporter.export
def interpolate(x, y, z, guess, grid, interp_type='linear', minimum_neighbors=3, gamma=0.25,
                kappa_star=5.052, ave_spacing=None, search_radius=None, rbf_func='linear',rbf_smooth=0):
    r"""Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: float
        x coordinate
    y: float
        y coordinate
    z: float
        observation value (or deviation)
    guess:(M, N) ndarray
        a first guess grid with the same dimensions as the grid variable
    grid: tuple
        a defined grid with the x and y points
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from Scipy.interpolate.
        2) "natural_neighbor", "barnes", or "cressman" from Metpy.mapping .
        Default "linear".
    minimum_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    ave_spacing: float
        The average spacing between the stations.
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See scipy.interpolate.Rbf for more
        information.
   rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    grid_x: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the x dimension
    grid_y: (N, 2) ndarray
        Meshgrid for the resulting interpolation in the y dimension ndarray
    img: (M, N) ndarray
        2-dimensional array representing the interpolated values for each grid.
    """

    grid_x, grid_y = grid

    if interp_type in ['linear', 'nearest', 'cubic']:
        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, z, (grid_x, grid_y), method=interp_type)

    elif interp_type == 'natural_neighbor':
        img = interpolation.natural_neighbor(x, y, z, grid_x, grid_y)

    elif interp_type in ['cressman', 'barnes']:
        if ave_spacing is None:
            ave_spacing = calc_avg_spacing(x, y, filter=False)
        #ave_spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))
        
        if search_radius is None:
            search_radius = ave_spacing

        if interp_type == 'cressman':

            img = interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius,
                                                 min_neighbors=minimum_neighbors,
                                                 kind=interp_type)

        elif interp_type == 'barnes':
            
        
            kappa = calc_kappa(ave_spacing, kappa_star)
            img = guess + interpolation.inverse_distance(x, y, z, grid_x, grid_y, search_radius,
                                                 gamma, kappa, min_neighbors=minimum_neighbors,
                                                 kind=interp_type)

    elif interp_type == 'rbf':

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension for observations.
        h = np.zeros((len(x)))

        rbfi = Rbf(x, y, h, z, function=rbf_func, smooth=rbf_smooth)

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension grid cell position.
        hi = np.zeros(grid_x.shape)
        img = rbfi(grid_x, grid_y, hi)

    else:
        raise ValueError('Interpolation option not available. '
                         'Try: linear, nearest, cubic, natural_neighbor, '
                         'barnes, cressman, rbf')

    return grid_x, grid_y, img
