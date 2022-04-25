#! /usr/bin/env python

import numpy as np
import scipy.linalg as la
from libdmet_solid.routine import slater
from libdmet_solid.utils.misc import mdot, max_abs, read_poscar, load_h5
from libdmet_solid.system import lattice
from libdmet_solid.utils.plot import eval_spin_corr_func_lo

cell = read_poscar(fname="/scratch/global/zhcui/proj/cuprate/mf/CaCuO2/SC/new_structure/CCO-2x2-frac.vasp")
cell.basis = 'def2-svp-no-diffuse'
kmesh = [4, 4, 2]
cell.spin = 0
cell.verbose = 5
cell.max_memory = 250000
cell.precision = 1e-11
cell.build()
aoind = cell.aoslice_by_atom()

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

print (Lat.cells)

Rs = []
cells = []
for i, x in enumerate(Lat.cells):
    if x[0] == 0 and x[2] == 0:
        cells.append(x)
        Rs.append(i)

print (cells)
print (Rs)

rdm1 = np.load("rdm1_0.npy")
basis_0 = np.load("basis.npy")
CuO2_idx = np.load("imp_idx_CuO2.npy") 
Cu0_idx = np.load("Cu0_idx.npy") 
Cu1_idx = np.load("Cu1_idx.npy") 
Cu2_idx = np.load("Cu2_idx.npy") 
Cu3_idx = np.load("Cu3_idx.npy") 

idx0 = Cu0_idx

for R in Rs:
    for idxR in [Cu0_idx, Cu2_idx]:
        print ("*" * 79)
        print (R, idxR)

        basis_R = slater.get_emb_basis_other_cell(Lat, basis_0, R)

        # 0 th problem
        C_0_0 = basis_0[:, 0][:, idx0]
        C_0_R = basis_0[:, R][:, idxR]

        # R th problem
        C_R_0 = basis_R[:, 0][:, idx0]
        C_R_R = basis_R[:, R][:, idxR]

        rdm1_00 = np.einsum('spi, sij, sqj -> spq', C_0_0, rdm1, C_0_0, optimize=True)
        rdm1_RR = np.einsum('spi, sij, sqj -> spq', C_R_R, rdm1, C_R_R, optimize=True)
        
        rdm1_R_R0 = np.einsum('spi, sij, sqj -> spq', C_R_R, rdm1, C_R_0, optimize=True)
        rdm1_0_R0 = np.einsum('spi, sij, sqj -> spq', C_0_R, rdm1, C_0_0, optimize=True)
        rdm1_R0 = (rdm1_0_R0 + rdm1_R_R0) * 0.5
        
        rdm1_a, rdm1_b = rdm1_R0
                
        print ("rdm2_aa")
        rdm2_aa = load_h5("rdm2_aa_0.h5")
        rdm2_RR00_aa  = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_aa, 
                                    C_R_R[0], C_R_R[0], C_R_0[0], C_R_0[0], optimize=True)
        rdm2_RR00_aa += np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_aa, 
                                    C_0_R[0], C_0_R[0], C_0_0[0], C_0_0[0], optimize=True)
        rdm2_aa = None
        rdm2_RR00_aa *= 0.5
        rdm2_RR00_aa += np.einsum('qp, sr -> pqrs', rdm1_RR[0], rdm1_00[0])
        rdm2_RR00_aa -= np.einsum('ps, qr -> pqrs', rdm1_R0[0], rdm1_R0[0])

        print ("rdm2_bb")
        rdm2_bb = load_h5("rdm2_bb_0.h5")
        rdm2_RR00_bb  = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_bb, 
                                    C_R_R[1], C_R_R[1], C_R_0[1], C_R_0[1], optimize=True)
        rdm2_RR00_bb += np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_bb, 
                                    C_0_R[1], C_0_R[1], C_0_0[1], C_0_0[1], optimize=True)
        rdm2_bb = None
        rdm2_RR00_bb *= 0.5
        rdm2_RR00_bb += np.einsum('qp, sr -> pqrs', rdm1_RR[1], rdm1_00[1])
        rdm2_RR00_bb -= np.einsum('ps, qr -> pqrs', rdm1_R0[1], rdm1_R0[1])

        print ("rdm2_ab")
        rdm2_ab = load_h5("rdm2_ab_0.h5")
        rdm2_RR00_ab  = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_ab, 
                                    C_R_R[0], C_R_R[0], C_R_0[1], C_R_0[1], optimize=True)
        rdm2_RR00_ab += np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_ab, 
                                    C_0_R[0], C_0_R[0], C_0_0[1], C_0_0[1], optimize=True)
        rdm2_RR00_ab *= 0.5
        rdm2_RR00_ab += np.einsum('qp, sr -> pqrs', rdm1_RR[0], rdm1_00[1])

        print ("rdm2_ba")
        rdm2_RR00_ba  = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_ab.transpose(2, 3, 0, 1), 
                                    C_R_R[1], C_R_R[1], C_R_0[0], C_R_0[0], optimize=True)
        rdm2_RR00_ba += np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2_ab.transpose(2, 3, 0, 1), 
                                    C_0_R[1], C_0_R[1], C_0_0[0], C_0_0[0], optimize=True)
        rdm2_RR00_ba *= 0.5
        rdm2_ab = None
        rdm2_RR00_ba += np.einsum('qp, sr -> pqrs', rdm1_RR[1], rdm1_00[0])

        print ("S1")

        norb = rdm1_R0.shape[-1]
        if R == 0 and max_abs(idx0 - idxR) < 1e-10:
            delta = np.eye(norb)
        else:
            delta = np.zeros((norb, norb))

        S_prod  = np.einsum('ij, ij ->', rdm1_a, delta, optimize=True)
        S_prod += np.einsum('ij, ij ->', rdm1_b, delta, optimize=True)
        S_prod *= 0.25

        print ("Sz")
        # Az
        S_corr  = 0.25 * np.einsum("iijj ->", rdm2_RR00_aa, optimize=True)
        S_corr += 0.25 * np.einsum("iijj ->", rdm2_RR00_bb, optimize=True)
        S_corr -= 0.25 * np.einsum("iijj ->", rdm2_RR00_ab, optimize=True)
        S_corr -= 0.25 * np.einsum("iijj ->", rdm2_RR00_ba, optimize=True)

        S = S_prod + S_corr

        print ("prod")
        print (S_prod)
        print ("corr")
        print (S_corr)
        print ("total")
        print (S)

        print ("*" * 79)
