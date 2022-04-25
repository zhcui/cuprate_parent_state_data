#!/usr/bin/env python

"""
CCO using 2x2 cell. calculate cumulant part of rdm2.
"""

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import libdmet.utils.logger as log
fout = open("dmet_%s.out"%rank, 'w')
log.stdout = fout

import os, sys
import time
import numpy as np
import scipy.linalg as la
from pyscf.pbc import gto, scf, df, dft, cc
from pyscf import lib
from pyscf.pbc.lib import chkfile
from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, read_poscar
from libdmet.basis_transform import make_basis
from libdmet.routine import pbc_helper as pbc_hp
from libdmet.routine import slater
import libdmet.dmet.Hubbard as dmet

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG1"

start = time.time()

cell = read_poscar(fname="../CCO-2x2-frac.vasp")
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

gdf_fname = '../gdf_ints_CCO_svp_442.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = 'def2-svp-ri'
gdf.linear_dep_threshold = 0.0
if not os.path.isfile(gdf_fname):
    gdf.build()

kmf = scf.KUHF(cell, kpts, exxdiv=None).density_fit()
kmf.exxdiv = None

kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-11
chk_fname = './CCO_UHF.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 150

data = chkfile.load("../CCO_UHF.chk", "scf")
kmf.__dict__.update(data)

kmf = make_basis.symmetrize_kmf(kmf, Lat)
ovlp = Lat.symmetrize_lo(np.load("../ovlp.npy"))
hcore = Lat.symmetrize_lo(np.load("../hcore.npy"))
rdm1 = np.asarray(kmf.make_rdm1())
rdm1 = Lat.symmetrize_lo(rdm1)

kmf.get_hcore = lambda *args: hcore
kmf.get_ovlp = lambda *args: ovlp

import copy 
from libdmet.utils.plot import get_dos
from libdmet.utils.misc import kdot, mdot
from libdmet.lo.iao import get_idx, get_idx_each_atom, \
        get_idx_each_orbital
from libdmet.basis_transform import make_basis
from libdmet.lo.iao import reference_mol, get_labels, get_idx_each
from libdmet.lo import pywannier90
from libdmet.basis_transform.eri_transform import get_unit_eri
from libdmet.lo.lowdin import check_orthogonal, check_orthonormal, \
        check_span_same_space

from libdmet.routine.mfd import assignocc 
from libdmet.utils import plot

minao = 'def2-svp-atom-iao' 

pmol = reference_mol(cell, minao=minao)
basis = pmol._basis

print ("basis")
for lab, bas in basis.items():
    print (lab)
    for b in bas:
        print (b)

basis_val = {}
basis_core = {}

minao_val = 'def2-svp-atom-iao-val' 
pmol_val = pmol.copy()
pmol_val.basis = minao_val
pmol_val.build()
basis_val["Cu"] = copy.deepcopy(pmol_val._basis["Cu"])
basis_val["O1"] = copy.deepcopy(pmol_val._basis["O1"])
basis_val["Ca"] = copy.deepcopy(pmol_val._basis["Ca"])

pmol_val = pmol.copy()
pmol_val.basis = basis_val
pmol_val.build()

val_labels = pmol_val.ao_labels()
for i in range(len(val_labels)):
    val_labels[i] = val_labels[i].replace("Ca 1s", "Ca 4s")
    val_labels[i] = val_labels[i].replace("Cu 1s", "Cu 4s")
    val_labels[i] = val_labels[i].replace("Cu 2p", "Cu 4p")
pmol_val.ao_labels = lambda *args: val_labels
print ("Valence:")
print (len(pmol_val.ao_labels()))
print (pmol_val.ao_labels())

# CORE
minao_core = 'def2-svp-atom-iao-core' #{"Cu": "def2-svp-atom-iao-core", "O1": "def2-svp-atom-iao-core"}
pmol_core = pmol.copy()
pmol_core.basis = minao_core
pmol_core.build()
basis_core["Cu"] = copy.deepcopy(pmol_core._basis["Cu"])
basis_core["O1"] = copy.deepcopy(pmol_core._basis["O1"])
basis_core["Ca"] = copy.deepcopy(pmol_core._basis["Ca"])

pmol_core = pmol.copy()
pmol_core.basis = basis_core
pmol_core.build()
core_labels = pmol_core.ao_labels()

print ("Core:")
print (len(pmol_core.ao_labels()))
print (pmol_core.ao_labels())


ncore = len(pmol_core.ao_labels())
nval = pmol_val.nao_nr()
nvirt = cell.nao_nr() - ncore - nval
Lat.set_val_virt_core(nval, nvirt, ncore)
    
mo_coeff = np.asarray(kmf.mo_coeff)
mo_occ = np.asarray(kmf.mo_occ)

# First construct IAO and PAO.
if rank == 0:
    if os.path.exists("C_ao_iao.npy"):
        C_ao_iao = np.load("C_ao_iao.npy")
        C_ao_iao_core = np.load("C_ao_iao_core.npy")
        C_ao_iao_xcore = np.load("C_ao_iao_xcore.npy")
    else:
        C_ao_iao, C_ao_iao_val, C_ao_iao_virt, C_ao_iao_core \
                = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True, \
                pmol_val=pmol_val, pmol_core=pmol_core, tol=1e-9)
        assert nval == C_ao_iao_val.shape[-1]

        print ("ncore: ", ncore)
        print ("nval: ", nval)
        print ("nvirt: ", nvirt)

        Lat.check_lo(C_ao_iao_core)
        Lat.check_lo(C_ao_iao_val)
        C_ao_iao_core = Lat.symmetrize_lo(C_ao_iao_core)
        C_ao_iao_val = Lat.symmetrize_lo(C_ao_iao_val)
        #C_ao_iao_virt = Lat.symmetrize_lo(C_ao_iao_virt)
        C_ao_iao_xcore = make_basis.tile_C_ao_iao(C_ao_iao_val, C_virt=C_ao_iao_virt, C_core=None)
        C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
        

        print ("check IAO")
        print ("orthonormal")
        print (check_orthonormal(C_ao_iao_core,  ovlp, tol=1e-9))
        print (check_orthonormal(C_ao_iao_val,  ovlp, tol=1e-9))
        #print (check_orthonormal(C_ao_iao_virt,  ovlp, tol=1e-9))
        print ("val core orth")
        print (check_orthogonal(C_ao_iao_val, C_ao_iao_core, ovlp, tol=1e-9))
        #print ("virt core orth")
        #print (check_orthogonal(C_ao_iao_virt, C_ao_iao_core, ovlp, tol=1e-9))
        #print ("val virt orth")
        #print (check_orthogonal(C_ao_iao_val, C_ao_iao_virt, ovlp, tol=1e-9))
        print ("core span same space")
        print (check_span_same_space(C_ao_iao_core, mo_coeff[:, :, :, :ncore], ovlp, tol=1e-9))
        print ("non-core span same space")
        print (check_span_same_space(C_ao_iao_xcore, mo_coeff[:, :, :, ncore:], ovlp,
            tol=1e-9))
        print ("check imag")
        print (Lat.check_lo(C_ao_iao))
        print (Lat.check_lo(C_ao_iao_xcore))
        print ("finish check IAO")
        
        np.save("C_ao_iao.npy", C_ao_iao)
        np.save("C_ao_iao_xcore.npy", C_ao_iao_xcore)
        np.save("C_ao_iao_core.npy", C_ao_iao_core)
comm.Barrier()
C_ao_iao = np.load("C_ao_iao.npy")
C_ao_iao_core = np.load("C_ao_iao_core.npy")
C_ao_iao_xcore = np.load("C_ao_iao_xcore.npy")

# ***********************
# DMET
# ***********************

if rank == 0:
    if os.path.exists("vj_core.npy"):
        rdm1_core = np.load("rdm1_core.npy") 
        rdm1_xcore = np.load("rdm1_xcore.npy") 
        vj_core = np.load("vj_core.npy")
        vk_core = np.load("vk_core.npy")
        vj_ao = np.load("vj_ao.npy")
        vk_ao = np.load("vk_ao.npy")
    else:
        rdm1_core = Lat.symmetrize_lo(kmf.make_rdm1(mo_coeff[:, :, :, :ncore], mo_occ[:, :, :ncore]))
        rdm1_xcore = Lat.symmetrize_lo(kmf.make_rdm1(mo_coeff[:, :, :, ncore:], mo_occ[:, :, ncore:]))
        np.save("rdm1_core.npy", rdm1_core)
        np.save("rdm1_xcore.npy", rdm1_xcore)
        
        vj_core, vk_core = kmf.get_jk(cell, rdm1_core)
        np.save("vj_core.npy", vj_core)
        np.save("vk_core.npy", vk_core)
         
        vj_ao, vk_ao = kmf.get_jk(cell, rdm1)
        np.save("vj_ao.npy", vj_ao)
        np.save("vk_ao.npy", vk_ao)

comm.Barrier()
rdm1_core = np.load("rdm1_core.npy") 
rdm1_xcore = np.load("rdm1_xcore.npy") 
vj_core = np.load("vj_core.npy")
vk_core = np.load("vk_core.npy")
vj_ao = np.load("vj_ao.npy")
vk_ao = np.load("vk_ao.npy")

vj_xcore = vj_ao - vj_core
vk_xcore = vk_ao - vk_core
veff_core = vj_core[0] + vj_core[1] - vk_core

vj_xcore = Lat.symmetrize_lo(vj_xcore)
vk_xcore = Lat.symmetrize_lo(vk_xcore)
veff_core = Lat.symmetrize_lo(veff_core)

hcore_xcore = hcore + veff_core
veff_xcore = vj_xcore[0] + vj_xcore[1] - vk_xcore

veff_ao = vj_ao[0] + vj_ao[1] - vk_ao
#E = np.einsum('skij, skji -> ', (hcore + veff_ao * 0.5), rdm1) / nkpts + kmf.energy_nuc()
E_core = np.einsum('skij, skji ->', (hcore + veff_core * 0.5), rdm1_core) / nkpts
if E_core.imag < 1e-10:
    E_core = E_core.real
E_xcore = np.einsum('skij, skji ->', (hcore + veff_core + veff_xcore * 0.5), rdm1_xcore) / nkpts
E_nuc = kmf.energy_nuc()

print ("Emf")
print (kmf.e_tot)
print ("E_core + E_xcore + E_nuc")
print (E_core + E_xcore + E_nuc)
print ("E_core")
print (E_core)
print ("E_xcore")
print (E_xcore)
print ("E_nuc")
print (E_nuc)

veff_core = vj_core[0] + vj_core[1] - vk_core
hcore_xcore = hcore + veff_core
#veff_xcore = vj_xcore[0] + vj_xcore[1] - vk_xcore

lo_labels = get_labels(cell, minao=None, full_virt=False, B2_labels=pmol_val.ao_labels(), \
        core_labels=pmol_core.ao_labels())[0]
idx_spec = get_idx_each(cell, labels=lo_labels, kind='atom')
idx_spec_val = get_idx_each(cell, labels=pmol_val.ao_labels(), kind='atom')
idx_atom = get_idx_each(cell, labels=lo_labels, kind='id atom')
idx_orb = get_idx_each(cell, labels=lo_labels, kind='id atom nl')
idx_all = get_idx_each(cell, labels=lo_labels, kind='all')

np.save("Cu0_idx.npy", idx_atom["0 Cu"])
np.save("Cu1_idx.npy", idx_atom["1 Cu"])
np.save("Cu2_idx.npy", idx_atom["2 Cu"])
np.save("Cu3_idx.npy", idx_atom["3 Cu"])

C_ao_lo = C_ao_iao_xcore
NCORE = ncore
NLO = C_ao_iao_xcore.shape[-1]
nlo = NLO

cell_lo = cell.copy()
cell_lo.nelectron = cell_lo.nelectron - NCORE * 2
cell_lo.nao_nr = lambda *args: NLO
cell_lo.build()

Lat = lattice.Lattice(cell_lo, kmesh)

Cu_idx = idx_spec["Cu"]
O1_idx = idx_spec["O1"]
Ca_idx = idx_spec["Ca"]
Cu_O1_idx = np.append(Cu_idx, O1_idx)
ion_idx = Ca_idx

Cu_idx_val = idx_spec_val["Cu"]
O1_idx_val = idx_spec_val["O1"]
Ca_idx_val = idx_spec_val["Ca"]
Cu_O1_idx_val = np.sort(np.append(Cu_idx_val, O1_idx_val), kind='mergesort')
Cu_O1_idx_virt = np.sort([idx for idx in Cu_O1_idx if not idx in Cu_O1_idx_val], kind='mergesort')

ion_idx_val = np.sort(Ca_idx_val, kind='mergesort')
ion_idx_virt = np.sort([idx for idx in ion_idx if not idx in ion_idx_val], kind='mergesort')

idx_sp_orb = get_idx_each(cell_lo, labels=lo_labels, kind='atom nl')

idx_x = get_idx_each(cell, labels=lo_labels, kind='atom nlm')

tband_idx = list(idx_all["0 Cu 3dx2-y2"]) + list(idx_all["1 Cu 3dx2-y2"])\
          + list(idx_all["2 Cu 3dx2-y2"]) + list(idx_all["3 Cu 3dx2-y2"])\
          + list(idx_all["4 O1 2py   "])     + list(idx_all["5 O1 2px   "])\
          + list(idx_all["6 O1 2py   "])     + list(idx_all["7 O1 2px   "])\
          + list(idx_all["8 O1 2px   "])     + list(idx_all["9 O1 2py   "])\
          + list(idx_all["10 O1 2px   "])     + list(idx_all["11 O1 2py   "])

from libdmet import utils
if rank == 0:
    Lat.set_val_virt_core(Cu_O1_idx_val, Cu_O1_idx_virt, [])
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
            rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=(kmf.energy_nuc()+E_core).real)
    imp_idx_fit = utils.search_idx1d(tband_idx, Lat.imp_idx)
    print ("imp_idx_fit")
    print (imp_idx_fit)
    print (np.asarray(lo_labels)[imp_idx_fit])
else:
    Lat.set_val_virt_core(ion_idx_val, ion_idx_virt, [])
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
        rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=0.0)
    imp_idx_fit = None

np.save("imp_idx_CuO2.npy", Lat.imp_idx)

# system
Filling = [(cell_lo.nelectron) / float(Lat.nscsites*2.0),
           (cell_lo.nelectron) / float(Lat.nscsites*2.0)]
restricted = False
bogoliubov = False
int_bath = True
nscsites = Lat.nscsites
Mu = 0.0
last_dmu = 0.0 #-0.010858738431
beta = 1000.0

# DMET SCF control
MaxIter = 100
u_tol = 5.0e-5
E_tol = 1.0e-5
iter_tol = 4

# DIIS
adiis = lib.diis.DIIS()
adiis.space = 4
diis_start = 3
dc = dmet.FDiisContext(adiis.space)
trace_start = 1000

# solver and mu fit
cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-6, tol_normt=2e-5, max_memory=cell.max_memory)

ncas = Lat.nval * 2 + Lat.nvirt
nelecas = Lat.nval * 2
solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
            splitloc=False, MP2natorb=False, cisolver=cisolver, \
            mom_reorder=False, tmpDir="./tmp")
solver.name = 'dmrgci_%s'%rank

nelec_tol = 1.0e+6
delta = 0.01
step = 1.0
load_frecord = False

# vcor fit
imp_fit = False
emb_fit_iter = 300 # embedding fitting
full_fit_iter = 0
ytol = 1e-7
gtol = 5e-4 
CG_check = False

idx_range = np.append(Cu_O1_idx_val, Cu_O1_idx_virt).astype(np.int)
vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites, idx_range=idx_range)

# DMET main loop
E_old = 0.0
conv = False
history = dmet.IterHistory()
dVcor_per_ele = None
dmet.SolveImpHam_with_fitting.load("./frecord")

Mu, last_dmu, vcor_param, rhoEmb, basis, rhoImp, Lat_fock, rho = \
        np.load("../DMET/dmet_iter_5.npy", allow_pickle=True)
vcor.update(vcor_param)

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    
    log.section("\nSolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %s", Mu)
    
    comm.Barrier()
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    rho_k = res["rho_k"]
    
    Lat.mulliken_lo_R0(rho[:, 0], labels=np.asarray(lo_labels))
    
    log.section("\nConstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, \
            int_bath=int_bath, max_memory=cell.max_memory, t_reversal_symm=True, \
            orth=True)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)
    
    if rank == 0:
        np.save("basis.npy", basis)
        np.save("JK_core.npy", Lat.JK_core)
    else:
        np.save("basis_ion.npy", basis)
        np.save("JK_core_ion.npy", Lat.JK_core)

    log.section("\nSolving impurity problem\n")
    ci_args = {"restart": True, "fcc_name" : "fcc_%s.h5"%(rank)}
    solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
            "guess": dmet.foldRho_k(res["rho_k"], basis_k), \
            "basis": basis,
            "ci_args": ci_args}

    rhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, \
        basis, solver, \
        solver_args=solver_args, thrnelec=nelec_tol, \
        delta=delta, step=step, comm=comm)
    
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    
    ImpHam = None
    rdm2 = cisolver.make_rdm2(ao_repr=True, with_dm1=False)
    mo = solver.cas
    rdm2_aa = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2[0], mo[0], mo[0], mo[0], mo[0], optimize=True)
    rdm2_bb = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2[1], mo[1], mo[1], mo[1], mo[1], optimize=True)
    rdm2_ab = np.einsum('ijkl, pi, qj, rk, sl -> pqrs', rdm2[2], mo[0], mo[0], mo[1], mo[1], optimize=True)
    rdm2 = None

    from libdmet.utils import save_h5
    save_h5("rdm2_aa_%s.h5"%rank, rdm2_aa)
    save_h5("rdm2_bb_%s.h5"%rank, rdm2_bb)
    save_h5("rdm2_ab_%s.h5"%rank, rdm2_ab)
    
    np.save("rdm1_%s.npy"%rank, rhoEmb)
    
    exit()
