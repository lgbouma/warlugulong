import os, socket
from warlugulong import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'reports', 'figures')

LOCALDIR = os.path.join(os.path.expanduser('~'), 'local', 'warlugulong')
if not os.path.exists(LOCALDIR):
    os.mkdir(LOCALDIR)
