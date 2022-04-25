#!/usr/bin/env python

"""
CCO using AFM cell.
"""

import os, sys
import time
import numpy as np
import scipy.linalg as la
from pyscf.pbc import gto, scf, df, dft, cc
from pyscf import lib
from pyscf.pbc.lib import chkfile
from libdmet.basis_transform import make_basis
from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, read_poscar
import libdmet.utils.logger as log
from libdmet.routine import pbc_helper as pbc_hp

log.verbose = "DEBUG1"

start = time.time()

cell = read_poscar(fname="../../../../01_crystal_geometry/CCO/CCO-AFM-frac.vasp")
cell.basis = {'Cu':'gth-szv-molopt-sr', 'O1': 'gth-szv-molopt-sr',
              'Ca':'gth-szv-molopt-sr'}
cell.pseudo = 'gth-pbe'
nelec_dop = 0
kmesh = [1, 1, 1]
nkpts = np.prod(kmesh)
cell.spin = nkpts * 2
cell.verbose = 5
cell.max_memory = 120000
cell.precision = 1e-11
cell.build()
aoind = cell.aoslice_by_atom()


Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = '../AFM/gdf_ints.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = df.aug_etb(cell, beta=2.3, use_lval=True, l_val_set={'Cu':2,'O1':1})
#gdf.auxbasis = df.aug_etb(cell, use_lval=True, l_val_set={'Cu':2,'O1':1,'O2':1})
gdf.linear_dep_threshold = 0.0
if not os.path.isfile(gdf_fname):
    gdf.build()

kmf = scf.KUHF(cell, kpts, exxdiv=None).density_fit()
kmf.exxdiv = None

dm0 = kmf.get_init_guess()
Cu_3d = cell.search_ao_label("Cu 3dx2-y2")
Cu0_3d = Cu_3d[:1]
Cu1_3d = Cu_3d[1:]
dm0[0, :, Cu0_3d, Cu0_3d] *= 2.0 
dm0[0, :, Cu1_3d, Cu1_3d] *= 2.0 
dm0[1, :, Cu0_3d, Cu0_3d]  = 0.0 
dm0[1, :, Cu1_3d, Cu1_3d]  = 0.0 

log.info('mf afm guess dm_a - dm_b: %s', max_abs(dm0[0] - dm0[1]))

kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-11
chk_fname = './CCO_UHF.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 150

if os.path.isfile(chk_fname):
    data = chkfile.load(chk_fname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf.kernel(dm0=dm0)

Lat.analyze(kmf)

from pyscf.pbc import cc
mycc = cc.KUCCSD(kmf, frozen=12)
mycc.kernel()

print (mycc.e_tot)

