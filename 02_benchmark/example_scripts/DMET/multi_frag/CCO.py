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
from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, read_poscar
import libdmet.utils.logger as log
from libdmet.basis_transform import make_basis
from libdmet.routine import pbc_helper as pbc_hp
from libdmet.routine import slater

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG1"

start = time.time()

cell = read_poscar(fname="../01_crystal_geometry/CCO/CCO-AFM-frac.vasp")
cell.basis = {'Cu':'gth-szv-molopt-sr', 'O1': 'gth-szv-molopt-sr',
              'Ca':'gth-szv-molopt-sr'}
cell.pseudo = 'gth-pbe'
kmesh = [6, 6, 1]
cell.spin = 0
cell.verbose = 5
cell.max_memory = 120000
cell.precision = 1e-11
cell.build()
aoind = cell.aoslice_by_atom()

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = '../gdf_ints_szv_662.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = df.aug_etb(cell, beta=2.3, use_lval=True, l_val_set={'Cu':2,'O1':1})
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

minao = 'gth-szv-molopt-sr'
pmol = reference_mol(cell, minao=minao)
basis = pmol._basis

basis_val = {}
basis_core = {}

# Ca core is 3s and 3p
basis_val["Ca"] = copy.deepcopy(basis["Ca"])
basis_core["Ca"] = copy.deepcopy(basis["Ca"])
# filter out the 3s
for i in range(1, len(basis_val["Ca"][0])):
    basis_val ["Ca"][0][i].pop(1) # valence pop out the 3s
    basis_core["Ca"][0][i].pop(2) # core pop out the 4s
# then 3p
basis_val["Ca"].pop(1) # valence pop out the 3p

# Cu, no core
basis_val["Cu"] = copy.deepcopy(basis["Cu"])

# O1 
basis_val["O1"] = copy.deepcopy(basis["O1"][1:])
basis_core["O1"] = copy.deepcopy(basis["O1"][:1])

pmol_val = pmol.copy()
pmol_val.basis = basis_val
pmol_val.build()

print ("Valence:")
val_labels = pmol_val.ao_labels()
for i in range(len(val_labels)):
    val_labels[i] = val_labels[i].replace("Ca 3s", "Ca 4s")
pmol_val.ao_labels = lambda *args: val_labels
print (len(pmol_val.ao_labels()))
print (pmol_val.ao_labels())

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

# ***********************
# DMET
# ***********************

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
    
vj_xcore = vj_ao - vj_core
vk_xcore = vk_ao - vk_core

veff_core = vj_core[0] + vj_core[1] - vk_core
hcore_xcore = hcore + veff_core
veff_xcore = vj_xcore[0] + vj_xcore[1] - vk_xcore

veff_ao = vj_ao[0] + vj_ao[1] - vk_ao
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

C_ao_lo = C_ao_iao_xcore
NCORE = ncore
NLO = C_ao_iao_xcore.shape[-1]
nlo = NLO

cell_lo = cell.copy()
cell_lo.nelectron = cell_lo.nelectron - NCORE * 2
cell_lo.nao_nr = lambda *args: NLO
cell_lo.build()

Lat = lattice.Lattice(cell_lo, kmesh)
Lat_ion = lattice.Lattice(cell_lo, kmesh)

Cu_idx = idx_spec["Cu"]
O1_idx = idx_spec["O1"]
Ca_idx = idx_spec["Ca"]
Cu_O1_idx = np.append(Cu_idx, O1_idx)

Cu_idx_val = idx_spec_val["Cu"]
O1_idx_val = idx_spec_val["O1"]
Cu_O1_idx_val = np.sort(np.append(Cu_idx_val, O1_idx_val), kind='mergesort')
Cu_O1_idx_virt = np.sort([idx for idx in Cu_O1_idx if not idx in Cu_O1_idx_val], kind='mergesort')

idx_sp_orb = get_idx_each(cell_lo, labels=lo_labels, kind='atom nl')

idx_x = get_idx_each(cell, labels=lo_labels, kind='atom nlm')
Cu_3band = list(idx_x["Cu 4s"]) + list(idx_x["Cu 3dx2-y2"])

O_3band = list(idx_all["2 O1 2px   "]) + list(idx_all["3 O1 2px   "]) + list(idx_all["4 O1 2py   "]) + list(idx_all["5 O1 2py   "])
tband_idx = np.sort(np.array(Cu_3band + O_3band), kind='mergesort')

Lat.set_val_virt_core(Cu_O1_idx_val, Cu_O1_idx_virt, [])
Lat_ion.set_val_virt_core(Ca_idx, [], [])

import libdmet.dmet.Hubbard as dmet

Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
        rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=(kmf.energy_nuc()+E_core).real)
Lat_ion.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
        rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=0.0)

from libdmet import utils
imp_idx_fit = utils.search_idx1d(tband_idx, Lat.imp_idx)
print ("imp_idx_fit")
print (imp_idx_fit)
print (np.asarray(lo_labels)[imp_idx_fit])

# system
Filling = [(cell_lo.nelectron) / float(Lat.nscsites*2.0),
           (cell_lo.nelectron) / float(Lat.nscsites*2.0)]
restricted = False
bogoliubov = False
int_bath = True
nscsites = Lat.nscsites
Mu = 0.0
#last_dmu = -0.167868262294
last_dmu = 0.003438110157
beta = 1000.0
#beta = np.inf

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
cisolver_ion = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-6, tol_normt=2e-5, max_memory=cell.max_memory)

# froze Cu 4f -> 4*7 = 28, O 3d -> 8 * 5 = 40
ncas = Lat.nval * 2 + Lat.nvirt
nelecas = Lat.nval * 2
solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
            splitloc=False, MP2natorb=False, cisolver=cisolver, \
            mom_reorder=False, tmpDir="./tmp")

ncas_ion = Lat_ion.nval * 2 + Lat_ion.nvirt
nelecas_ion = Lat_ion.nval * 2
solver_ion = dmet.impurity_solver.CASCI(ncas=ncas_ion, nelecas=nelecas_ion, \
            splitloc=False, MP2natorb=False, cisolver=cisolver_ion, \
            mom_reorder=False, tmpDir="./tmp")

nelec_tol = 1.0e-6
delta = 0.1
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

Cu_0_idx = idx_atom["0 Cu"]
Cu_1_idx = idx_atom["1 Cu"]
Cu_0_3d = idx_orb["0 Cu 3d"]
Cu_1_3d = idx_orb["1 Cu 3d"]
Cu0_mesh = np.ix_(Cu_0_idx, Cu_0_idx)
Cu1_mesh = np.ix_(Cu_1_idx, Cu_1_idx)

def get_rho_m(rdm1_R0, Cu_0_idx, Cu_1_idx):
    rho_0_a = rdm1_R0[0, Cu_0_idx, Cu_0_idx].sum()
    rho_0_b = rdm1_R0[1, Cu_0_idx, Cu_0_idx].sum()
    rho_1_a = rdm1_R0[0, Cu_1_idx, Cu_1_idx].sum()
    rho_1_b = rdm1_R0[1, Cu_1_idx, Cu_1_idx].sum()
    rho_0 = rho_0_a + rho_0_b
    rho_1 = rho_1_a + rho_1_b
    m_0 = rho_0_a - rho_0_b
    m_1 = rho_1_a - rho_1_b
    print ("rho_0", rho_0)
    print ("rho_1", rho_1)
    print ("m_0", m_0)
    print ("m_1", m_1)
    rho_ave = 0.5 * (rho_0 + rho_1)
    m_ave = 0.5 * (abs(m_0) + abs(m_1))
    return rho_ave, m_ave

def get_Cu_d(rdm1_R0, Cu_0_3d, Cu_1_3d):
    mesh_0 = np.ix_(Cu_0_3d, Cu_0_3d)
    mesh_1 = np.ix_(Cu_1_3d, Cu_1_3d)
    rdm1_0_a = rdm1_R0[0][mesh_0]
    rdm1_0_b = rdm1_R0[1][mesh_0]
    rdm1_0 = np.asarray((rdm1_0_a, rdm1_0_b))
    rdm1_1_a = rdm1_R0[0][mesh_1]
    rdm1_1_b = rdm1_R0[1][mesh_1]
    rdm1_1 = np.asarray((rdm1_1_a, rdm1_1_b))
    return rdm1_0, rdm1_1

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    
    log.section("\nSolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %s", Mu)
    
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    rho_k = res["rho_k"]
    
    rho_Cu, m_Cu = get_rho_m(rho[:, 0], Cu_0_idx, Cu_1_idx)
    print ("rho Cu (mf):", rho_Cu)
    print ("m Cu (mf):", m_Cu)
    rdm1_0, rdm1_1 = get_Cu_d(rho[:, 0], Cu_0_3d, Cu_1_3d)
    print ("rdm1 for 0th Cu:\n%s"%rdm1_0)
    print ("rdm1 for 1th Cu:\n%s"%rdm1_1)
    
    # Ca
    log.section("\nConstructing impurity problem\n")
    ImpHam_ion, H1e_ion, basis_ion = dmet.ConstructImpHam(Lat_ion, rho, vcor, matching=False, \
            int_bath=int_bath, max_memory=cell.max_memory, t_reversal_symm=True, \
            orth=True)
    ImpHam_ion = dmet.apply_dmu(Lat_ion, ImpHam_ion, basis_ion, last_dmu)
    basis_k_ion = Lat_ion.R2k_basis(basis_ion)

    log.section("\nSolving impurity problem\n")
    ci_args_ion = {"restart": False}
    solver_args_ion = {"nelec": (Lat_ion.ncore+Lat_ion.nval)*2, \
            "guess": dmet.foldRho_k(res["rho_k"], basis_k_ion), \
            "basis": basis_ion,
            "ci_args": ci_args_ion}
    
    # Cu O
    log.section("\nConstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, \
            int_bath=int_bath, max_memory=cell.max_memory, t_reversal_symm=True, \
            orth=True)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)

    log.section("\nSolving impurity problem\n")
    ci_args = {"restart": True}
    solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
            "guess": dmet.foldRho_k(res["rho_k"], basis_k), \
            "basis": basis,
            "ci_args": ci_args}

    basis_col = [basis, basis_ion]
    Lat_col = [Lat, Lat_ion]

    rhoEmb_col, EnergyEmb_col, ImpHam_col, dmu = \
        dmet.SolveImpHam_with_fitting(Lat_col, Filling, [ImpHam, ImpHam_ion], \
        basis_col, [solver, solver_ion], \
        solver_args=[solver_args, solver_args_ion], thrnelec=nelec_tol, \
        delta=delta, step=step)
    
    veff = slater.get_veff_from_rdm1_emb(Lat_col, rhoEmb_col, basis_col)
    
    ImpHam, ImpHam_ion = ImpHam_col
    rhoEmb, rhoEmb_ion = rhoEmb_col
    EnergyEmb, EnergyEmb_ion = EnergyEmb_col
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    
    np.save("basis.npy", basis)
    np.save("JK_core.npy", Lat.JK_core)
    E_hf_solver = slater.get_E_dmet_HF(basis, Lat, ImpHam, last_dmu, solver)
    log.result("E(HF solver) = %20.12f", E_hf_solver)
    
    print (rhoEmb.shape)
    print (basis.shape)
    print (ImpHam.H1["cd"].shape)

    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args, veff=veff)
    EnergyImp *= nscsites
    log.result("last_dmu = %20.12f", last_dmu)
    log.result("E(DMET) = %20.12f", EnergyImp)
    
    rho_Cu, m_Cu = get_rho_m(rhoImp, Cu_0_idx, Cu_1_idx)
    print ("rho Cu (corr):", rho_Cu)
    print ("m Cu (corr):", m_Cu)
    rdm1_0, rdm1_1 = get_Cu_d(rhoImp, Cu_0_3d, Cu_1_3d)
    print ("rdm1 for 0th Cu:\n%s"%rdm1_0)
    print ("rdm1 for 1th Cu:\n%s"%rdm1_1)
    
    Lat.mulliken_lo_R0(rhoImp, labels=np.asarray(lo_labels)[Lat.imp_idx])


    np.save("basis_ion.npy", basis_ion)
    np.save("JK_core_ion.npy", Lat_ion.JK_core)
    E_hf_solver_ion = slater.get_E_dmet_HF(basis_ion, Lat_ion, ImpHam_ion, last_dmu, solver_ion)
    log.result("E(HF solver ion) = %20.12f", E_hf_solver_ion)

    rhoImp_ion, EnergyImp_ion, nelecImp_ion = \
            dmet.transformResults(rhoEmb_ion, EnergyEmb_ion, basis_ion, ImpHam_ion, H1e_ion, \
            lattice=Lat_ion, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver_ion, solver_args=solver_args_ion, veff=veff)
    EnergyImp_ion *= nscsites
    log.result("E(DMET ion) = %20.12f", EnergyImp_ion)
    
    EnergyImp += EnergyImp_ion

    # DUMP results:
    dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
            Lat.getFock(kspace=False), rho], dtype=object)
    np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)
    
    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter, method='CG', \
            imp_fit=imp_fit, ytol=ytol, gtol=gtol, CG_check=CG_check, num_cg_steps=200, \
            max_stepsize=0.001, remove_diag_grad=False, imp_idx=imp_idx_fit)

    if iter >= trace_start:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)

    dE = EnergyImp - E_old
    E_old = EnergyImp 
    
    if iter >= diis_start:
        pvcor = adiis.update(vcor_new.param)
        dc.nDim = adiis.get_num_vec()
    else:
        pvcor = vcor_new.param
    
    dVcor_per_ele = max_abs(pvcor - vcor.param)
    vcor.update(pvcor)
    log.result("Trace of vcor: %s", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
    
    history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
    history.write_table()
    
    # ZHC NOTE convergence criterion
    if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
        conv = True
        break

if conv:
    log.result("DMET converge.")
else:
    log.result("DMET does not converge.")

