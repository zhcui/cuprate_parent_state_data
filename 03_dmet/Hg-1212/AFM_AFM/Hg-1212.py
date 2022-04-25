#!/usr/bin/env python

"""
HF Hg1212 using 2x2 cell.
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

cell = read_poscar(fname="./HBCO1212-2x2-red.vasp")
cell.basis = "def2-svp-no-diffuse"
cell.ecp = {"Hg": 'def2-svp-no-diffuse', "Ba": 'def2-svp-no-diffuse'}
nelec_dop = 0
kmesh = [4, 4, 1]
nkpts = np.prod(kmesh)
cell.spin = 0
cell.verbose = 5
cell.max_memory = 350000
cell.precision = 1e-12
cell.build()
aoind = cell.aoslice_by_atom()

# first layer 
Cu_A1 = cell.search_ao_label("Cu1 3dx2-y2")
Cu_B1 = cell.search_ao_label("Cu2 3dx2-y2")
# second layer 
Cu_A2 = cell.search_ao_label("Cu4 3dx2-y2")
Cu_B2 = cell.search_ao_label("Cu3 3dx2-y2")

Cu_3d_A = np.hstack((Cu_A1, Cu_A2))
Cu_3d_B = np.hstack((Cu_B1, Cu_B2))

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = './gdf_ints_svp_441.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = 'def2-svp-ri'
# default is [25, 25, 43]
# 1.2x
gdf.mesh = np.array([31, 31, 51])
gdf.linear_dep_threshold = 1e-11
if not os.path.isfile(gdf_fname):
    gdf.build(keep_aux_dim=True)

kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, \
        space_group_symmetry=True, time_reversal_symmetry=True)

kmf_symm = scf.KUHF(cell, kpts_symm, exxdiv=None).density_fit()
kmf_symm.exxdiv = None
from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV

#data = chkfile.load("../../../def2-svp_all/UHF/AA_LO/HBCO_UHF.chk", 'scf')
#kmf_symm.__dict__.update(data)
#dm0 = np.asarray(kmf_symm.make_rdm1())
dm0 = kmf_symm.get_init_guess(key='atom')
dm0[1] = dm0[0]

dm0[0, :, Cu_3d_A, Cu_3d_A]  *= 2.0 # 0 Cu, alpha is more
dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0 # 0 Cu, alpha is more
dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0 # 1 Cu, beta is less
dm0[1, :, Cu_3d_B, Cu_3d_B]  *= 2.0 # 2 Cu, beta is more

kmf_symm.with_df = gdf
kmf_symm.with_df._cderi = gdf_fname
kmf_symm.conv_tol = 1e-9
chk_fname = './HBCO_UHF.chk'
kmf_symm.chkfile = chk_fname
kmf_symm.diis_space = 15
kmf_symm.max_cycle = 150

kmf_symm.kernel(dm0=dm0)

kmf = pbc_hp.kmf_symm_(kmf_symm)

Lat.analyze(kmf, pre_orth_ao='SCF')

ovlp = np.asarray(kmf.get_ovlp())

np.save("hcore.npy", kmf.get_hcore())
np.save("ovlp.npy", kmf.get_ovlp())
np.save("fock.npy", kmf.get_fock())
np.save("rdm1.npy", kmf.make_rdm1())

