"""
Most of this calculation is ported from tests/sandbox_gala, which is direct
from the tutorials Adrian wrote up in his documentation.
"""
import os
from os.path import join
from astropy import units as u
import warlugulong.plotting as p
from warlugulong.paths import RESULTSDIR

clusterid = 'UCL'

plotdir = join(RESULTSDIR, f'visualize_{clusterid}')
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

p.plot_integration(
    clusterid, plotdir, n_steps=int(1e3), dt=-0.01*u.Myr, rv_method='fix',
    movie_durn=60*u.second
)

p.plot_integration(
    clusterid, plotdir, n_steps=int(2e3), dt=0.05*u.Myr, rv_method='fix',
    movie_durn=60*u.second
)

# takes 30 minutes to write(!!)
p.plot_integration(
    clusterid, plotdir, n_steps=int(20e3), dt=0.05*u.Myr, rv_method='fix',
    movie_durn=180*u.second
)
