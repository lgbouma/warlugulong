"""
plot_integration
"""
#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception

#############
## IMPORTS ##
#############

import os, collections, pickle
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from glob import glob
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict
from os.path import join

from numpy import array as nparr

from astropy.io import fits
from astropy import units as u
from astropy.table import Table

from astropy.coordinates import Galactocentric
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
_ = coord.galactocentric_frame_defaults.set('v4.0')

from aesthetic.plot import savefig, format_ax, set_style

from earhart.backintegrate import backintegrate
from earhart.physicalpositions import given_gaia_df_get_icrs_arr

from rudolf.helpers import get_ScoOB2_members

from warlugulong.paths import DATADIR, RESULTSDIR, LOCALDIR

import gala.dynamics as gd

def plot_integration(clusterid, plotdir, n_steps=int(2e3), dt=-0.05*u.Myr,
                     rv_method='fix', movie_durn=60*u.second):
    """
    Integrate the orbits of stars in a cluster, to see what happens.

    Args:
        clusterid: identifier string.

        n_steps: how many integration steps?

        dt: time step.  Negative means back-integration.

        rv_method: flag.  "fix" means set all the RVs to the mean cluster RV.
        "fix_{N}kms" (e.g., fix_2kms) sets the RVs to a normal distribution,
        with the mean cluster RV, and width set by N (km/s).

    For the neighborhood/field samples, we also require S/N>10 on the parallax
    measurement --- otherwise we get stars with negative parallaxes, which lead
    to erroneous distance measurements when defining the orbits and doing the
    back-integration.

    Key citations for the downstream code include a) Price-Whelan's `gala`
    code; b) Bovy 2015, who worked out the model for the galactic potential
    that is used in thisc calculation.
    """

    assert clusterid in ['ScoOB2', 'UCL', 'Orion']
    if clusterid == 'UCL':
        _df = get_ScoOB2_members()
    else:
        raise NotImplementedError

    rvkey = 'dr2_radial_velocity' if 'edr3' in clusterid else 'radial_velocity'
    get6cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', rvkey]

    # require 3d positions + 2d kinematics
    # also require S/N on parallax measurement
    # and apply whatever your base assumption is about the RVs
    if rv_method == 'fix':
        sdf = _df[_df.parallax_over_error > 10]

        sdf_6d = sdf[get6cols].dropna(axis=0)
        LOGINFO(f'Found N={len(sdf_6d)} {clusterid} members with 6d positions/kinematics.')
        med_rv = float(sdf_6d[rvkey].median())
        std_rv = float(sdf_6d[rvkey].std())

        LOGINFO(f'Fixing RV to {med_rv:.2f} km/s for all N={len(sdf)} 5+d members...')
        sdf[rvkey] = med_rv

    else:
        raise NotImplementedError

    # calculate things!
    icrs_arr = given_gaia_df_get_icrs_arr(sdf, zero_rv=0)

    times = np.arange(0, (n_steps+1)*dt.value, dt.value)*u.Myr
    orbits = backintegrate(icrs_arr, dt=dt, n_steps=n_steps)

    #
    # make the "orbit trace" plot
    #
    outpath = join(
        plotdir,
        f'orbit_trace_{clusterid}_{rv_method}_N{n_steps}_dt{dt}.png'
    )

    set_style()

    fig = orbits.plot()

    savefig(fig, outpath)
    plt.close('all')

    #
    # initial locations
    #
    outpath = join(
        plotdir,
        f'initial_locs_{clusterid}_{rv_method}_N{n_steps}_dt{dt}.png'
    )

    w = gd.PhaseSpacePosition(orbits[0,:].pos, orbits[0,:].vel)
    fig = w.plot(alpha=0.1)

    savefig(fig, outpath)
    plt.close('all')

    #
    # final locations
    #
    outpath = join(
        plotdir,
        f'final_locs_{clusterid}_{rv_method}_N{n_steps}_dt{dt}.png'
    )

    w = gd.PhaseSpacePosition(orbits[-1,:].pos, orbits[-1,:].vel)
    fig = w.plot(alpha=0.1)

    savefig(fig, outpath)
    plt.close('all')

    # make a movie
    outpath = join(
        plotdir,
        f'locs_{clusterid}_{rv_method}_N{n_steps}_dt{dt}.mp4'
    )
    outpath = outpath.replace(' ','_')

    if not os.path.exists(outpath):
        given_orbits_make_movie(orbits, times, outpath, movie_durn=movie_durn)
    else:
        LOGINFO(f'Found {outpath}, skipping.')


def given_orbits_make_movie(
    orbits, times, outpath,
    movie_durn=60*u.second # movie run-time (frame-rate is scaled accordingly)
):

    import matplotlib.animation as animation

    # set up axes and limits
    fig, axs = plt.subplots(figsize=(12,4), ncols=3)

    dx = np.max(orbits.x) - np.min(orbits.x)
    dy = np.max(orbits.y) - np.min(orbits.y)
    dz = np.max(orbits.z) - np.min(orbits.z)

    # axs0: Y vs X
    # axs1: Z vs X
    # axs1: Z vs Y
    axs[0].set_xlim([ (np.min(orbits.x)-0.1*dx).to(u.kpc).value, (np.max(orbits.x)+0.1*dx).to(u.kpc).value ])
    axs[0].set_ylim([ (np.min(orbits.y)-0.1*dy).to(u.kpc).value, (np.max(orbits.y)+0.1*dy).to(u.kpc).value ])
    axs[1].set_xlim([ (np.min(orbits.x)-0.1*dx).to(u.kpc).value, (np.max(orbits.x)+0.1*dx).to(u.kpc).value ])
    axs[1].set_ylim([ (np.min(orbits.z)-0.1*dz).to(u.kpc).value, (np.max(orbits.z)+0.1*dz).to(u.kpc).value ])
    axs[2].set_xlim([ (np.min(orbits.y)-0.1*dy).to(u.kpc).value, (np.max(orbits.y)+0.1*dy).to(u.kpc).value ])
    axs[2].set_ylim([ (np.min(orbits.z)-0.1*dz).to(u.kpc).value, (np.max(orbits.z)+0.1*dz).to(u.kpc).value ])

    axs[0].update({'xlabel': 'X [kpc]', 'ylabel': 'Y [kpc]'})
    axs[1].update({'xlabel': 'X [kpc]', 'ylabel': 'Z [kpc]'})
    axs[2].update({'xlabel': 'Y [kpc]', 'ylabel': 'Z [kpc]'})

    # get initial data to plot for first frame
    x = orbits.x[0,:].to(u.kpc).value
    y = orbits.y[0,:].to(u.kpc).value
    z = orbits.z[0,:].to(u.kpc).value

    numframes = orbits.shape[0]
    numpoints = orbits.shape[1]

    scat0 = axs[0].scatter(x, y, c='k', s=1, rasterized=True)
    scat1 = axs[1].scatter(x, z, c='k', s=1, rasterized=True)
    scat2 = axs[2].scatter(y, z, c='k', s=1, rasterized=True)

    txt = axs[0].text(0.03, 0.03, f't={times[0].to(u.Myr).value:.2f} Myr',
                      transform=axs[0].transAxes, ha='left',va='bottom',
                      color='k', fontsize='small')

    plt.tight_layout()

    LOGINFO(f'Beginning {outpath} creation...')

    frame_interval = (movie_durn / numframes).to(u.millisecond).value
    LOGINFO(f'Doing a movie {movie_durn} long.')
    LOGINFO(f'Frame interval is {frame_interval} milliseconds.')

    ani = animation.FuncAnimation(
        fig, # figure object which gets needed events
        update_plot, # function to call at each frame
        frames=range(numframes),
        interval=frame_interval,  # delay btwn frames (msec)
        fargs=(orbits, times, scat0, scat1, scat2, txt)
    )
    ani.save(outpath)
    LOGINFO(f'Wrote {outpath}')

def update_plot(i, orbits, times, scat0, scat1, scat2, txt):

    scat0.set_offsets(np.c_[orbits.x[i,:], orbits.y[i,:]])
    scat1.set_offsets(np.c_[orbits.x[i,:], orbits.z[i,:]])
    scat2.set_offsets(np.c_[orbits.y[i,:], orbits.z[i,:]])
    txt.set_text(f't={times[i].to(u.Myr).value:.2f} Myr')

    return scat0, scat1, scat2, txt,
