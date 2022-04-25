#!/usr/bin/env python

"""
CCO using 2x2 cell.
"""

import os, sys
import time
import numpy as np
import scipy.linalg as la
from pyscf.pbc import gto, scf, df, dft, cc
from pyscf import lib
from pyscf.pbc.lib import chkfile
from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, read_poscar
import libdmet.utils.logger as log
from libdmet.routine import pbc_helper as pbc_hp

log.verbose = "DEBUG1"

start = time.time()

cell = read_poscar(fname="../../../../../01_crystal_geometry/CCO/CCO-2x2-frac.vasp")
cell.basis = {'Cu': 'def2-svp-no-diffuse', 'O1': 'def2-svp-no-diffuse',
              'Ca': 'def2-svp-no-diffuse'}
kmesh = [4, 4, 2]
cell.spin = 0
cell.verbose = 5
cell.max_memory = 250000
cell.precision = 1e-11
cell.build()
aoind = cell.aoslice_by_atom()

aoind = cell.aoslice_by_atom()
Cu_3d = cell.search_ao_label("Cu 3dx2-y2")
Cu_3d_A = Cu_3d[[0, 1]]
Cu_3d_B = Cu_3d[[2, 3]]

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = '../gdf_ints_CCO_svp_442.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = 'def2-svp-ri'
gdf.linear_dep_threshold = 0.0
if not os.path.isfile(gdf_fname):
    gdf.build()

kmf = dft.KUKS(cell, kpts).density_fit()
kmf.xc = 'pbe0'
kmf.grids.level = 5
kmf.exxdiv = None

from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV
kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="fermi", tol=1e-12)

kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-10
chk_fname = './CCO_UHF.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 150

dm0 = np.asarray(kmf.get_init_guess(key='atom'))

dm0[0, :, Cu_3d_A, Cu_3d_A] *= 2.0 
dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0 
dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0 
dm0[1, :, Cu_3d_B, Cu_3d_B] *= 2.0 

#print (max_abs(dm0[0] - dm0[1]))
#Lat.mulliken_lo_R0(Lat.k2R(dm0)[:, 0], labels=np.asarray(cell.ao_labels()))

kmf.kernel(dm0=dm0)

Lat.analyze(kmf)

log.info('S^2 = %s, 2S+1 = %s' % kmf.spin_square())
np.save("hcore.npy", kmf.get_hcore())
np.save("ovlp.npy", kmf.get_ovlp())
np.save("fock.npy", kmf.get_fock())
ovlp = np.asarray(kmf.get_ovlp())

