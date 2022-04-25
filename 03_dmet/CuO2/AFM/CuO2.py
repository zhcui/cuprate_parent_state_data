#!/usr/bin/env python

"""
HF CuO2 using 2x2 cell.
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

cell = read_poscar(fname="./CuO2-2x2-frac.vasp")
cell.basis = {'Cu1':'def2-svp', 'Cu2': 'def2-svp', 'O1': 'def2-svp'}
nelec_dop = 0
kmesh = [4, 4, 2]
nkpts = np.prod(kmesh)
cell.spin = 0
cell.verbose = 5
cell.max_memory = 180000
cell.precision = 1e-11
cell.nelectron = (90 + 2 * 2) * 2
cell.build()
aoind = cell.aoslice_by_atom()

Cu_3d_A = cell.search_ao_label("Cu1 3dx2-y2")
Cu_3d_B = cell.search_ao_label("Cu2 3dx2-y2")

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = './gdf_ints_svp_442.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = {"Cu1": "def2-svp-ri", "Cu2": "def2-svp-ri", "O1": "def2-svp-ri"}

gdf.linear_dep_threshold = 0.0
if not os.path.isfile(gdf_fname):
    gdf.build()

kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, \
        space_group_symmetry=True,time_reversal_symmetry=True)

kmf_symm = scf.KUHF(cell, kpts_symm, exxdiv=None).density_fit()
kmf_symm.exxdiv = None
from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV

dm0 = np.asarray(kmf_symm.get_init_guess(key='atom'))
dm0[0, :, Cu_3d_A, Cu_3d_A] *= 2.0 
dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0 
dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0 
dm0[1, :, Cu_3d_B, Cu_3d_B] *= 2.0 

kmf_symm.with_df = gdf
kmf_symm.with_df._cderi = gdf_fname
kmf_symm.conv_tol = 1e-10
chk_fname = './CuO2_UHF.chk'
kmf_symm.chkfile = chk_fname
kmf_symm.diis_space = 15
kmf_symm.max_cycle = 150

kmf_symm.kernel(dm0=dm0)

kmf = pbc_hp.kmf_symm_(kmf_symm)

Lat.analyze(kmf)

ovlp = np.asarray(kmf.get_ovlp())

log.info('S^2 = %s, 2S+1 = %s' % kmf.spin_square())
np.save("hcore.npy", kmf.get_hcore())
np.save("ovlp.npy", kmf.get_ovlp())
np.save("fock.npy", kmf.get_fock())

