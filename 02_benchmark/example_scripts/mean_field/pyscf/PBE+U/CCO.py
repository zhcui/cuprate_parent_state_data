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
from libdmet.basis_transform import make_basis

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG1"

UVAL = 7.5

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
Cu_3d = cell.search_ao_label("Cu 3d.*")

Cu_3dx2y2 = cell.search_ao_label("Cu 3dx2-y2")
Cu_3d_A = Cu_3dx2y2[[0, 1]]
Cu_3d_B = Cu_3dx2y2[[2, 3]]

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

d_0 = cell.search_ao_label("Cu 3dxy")
d_1 = cell.search_ao_label("Cu 3dyz")
d_2 = cell.search_ao_label("Cu 3dz.")
d_3 = cell.search_ao_label("Cu 3dxz")
d_4 = cell.search_ao_label("Cu 3dx2-y2")
d_orbs = np.array((d_0, d_1, d_2, d_3, d_4))
assert d_orbs.shape == (5, 4)

C_ao_lo = np.zeros((2, nkpts, nao, nao), dtype=np.complex128)
for orbs in d_orbs:
    for idx in orbs:
        C_ao_lo[:, :, idx, idx] = 1.0

U_idx = ["Cu 3d"]
U_val = [UVAL]
kmf = pbc_hp.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, 
        C_ao_lo='meta-lowdin', pre_orth_ao='ANO').density_fit()
ovlp = kmf.get_ovlp()
lo_ovlp = make_basis.transform_h1_to_lo(ovlp, C_ao_lo[:, :, :, Cu_3d])

lo_ovlp = make_basis.transform_h1_to_lo(ovlp, C_ao_lo[:, :, :, Cu_3d])
for s in range(2):
    for k in range(nkpts):
        print (max_abs(np.diag(lo_ovlp[s, k]) - 1.0))

kmf = pbc_hp.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val, 
        C_ao_lo=C_ao_lo).density_fit()

kmf.exxdiv = None
kmf.xc = 'pbe'
kmf.grids.level = 5

from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV
kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="gaussian", tol=1e-14, fit_spin=False)

kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-10
chk_fname = './CCO_UPBEpU.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 150

dm0 = np.asarray(kmf.get_init_guess(key='atom'))

dm0[0, :, Cu_3d_A, Cu_3d_A] *= 2.0 
dm0[0, :, Cu_3d_B, Cu_3d_B]  = 0.0 
dm0[1, :, Cu_3d_A, Cu_3d_A]  = 0.0 
dm0[1, :, Cu_3d_B, Cu_3d_B] *= 2.0 

Lat.mulliken_lo_R0(Lat.k2R(dm0)[:, 0], labels=np.asarray(cell.ao_labels()))


kmf.kernel(dm0=dm0)

Lat.analyze(kmf, pre_orth_ao='def2-svp-atom')

log.info('S^2 = %s, 2S+1 = %s' % kmf.spin_square())
np.save("hcore.npy", kmf.get_hcore())
np.save("ovlp.npy", kmf.get_ovlp())
np.save("fock.npy", kmf.get_fock())
ovlp = np.asarray(kmf.get_ovlp())

