#!/usr/bin/env python

"""
HBCO using 2x2 cell.
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
from libdmet.routine import slater

log.verbose = "DEBUG1"
np.set_printoptions(3, linewidth=1000, suppress=True)

start = time.time()

cell = read_poscar(fname="../Hg1201_0.00_all.vasp")
cell.basis = {'Cu1':'def2-svp-no-diffuse', 'Cu2':'def2-svp-no-diffuse', 'O1': 'def2-svp-no-diffuse', 'O2': 'def2-svp-no-diffuse', \
        'Hg':'def2-svp-no-diffuse', 'Ba': 'def2-svp-no-diffuse'}
cell.ecp = {"Hg": 'def2-svp-no-diffuse', "Ba": 'def2-svp-no-diffuse'}
kmesh = [4, 4, 2]
nkpts = np.prod(kmesh)
cell.spin = 0 
cell.verbose = 5
cell.max_memory = 1080000
cell.precision = 1e-12
cell.build()

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = '../gdf_ints_svp_442.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = 'def2-svp-ri'
#gdf.linear_dep_threshold = 0.0
#if not os.path.isfile(gdf_fname):
#    gdf.build()

kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, \
        space_group_symmetry=True,time_reversal_symmetry=True)

kmf_symm = scf.KUHF(cell, kpts_symm, exxdiv=None).density_fit()
kmf_symm.exxdiv = None
from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV

kmf_symm.with_df = gdf
kmf_symm.with_df._cderi = gdf_fname
kmf_symm.conv_tol = 1e-11
chk_fname = './HBCO_UHF.chk'
kmf_symm.chkfile = chk_fname
kmf_symm.diis_space = 15
kmf_symm.max_cycle = 150

data = chkfile.load("../HBCO_UHF.chk", 'scf')
kmf_symm.__dict__.update(data)

kmf = kmf_symm.to_khf()
kmf.e_tot = kmf_symm.e_tot
kmf.kpts = kpts

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

C_ao_iao, C_ao_iao_val, C_ao_iao_virt = \
        make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True, tol=1e-9)

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
basis_val["Cu1"] = copy.deepcopy(pmol_val._basis["Cu1"])
basis_val["Cu2"] = copy.deepcopy(pmol_val._basis["Cu2"])
basis_val["O1"] = copy.deepcopy(pmol_val._basis["O1"])
basis_val["O2"] = copy.deepcopy(pmol_val._basis["O2"])
basis_val["Hg"] = copy.deepcopy(pmol_val._basis["Hg"])
basis_val["Ba"] = copy.deepcopy(pmol_val._basis["Ba"])

pmol_val = pmol.copy()
pmol_val.basis = basis_val
pmol_val.build()

val_labels = pmol_val.ao_labels()
for i in range(len(val_labels)):
    val_labels[i] = val_labels[i].replace("Ba 5s", "Ba 6s")
    #val_labels[i] = val_labels[i].replace("Ca 3s", "Ca 4s")
    val_labels[i] = val_labels[i].replace("Hg 5s", "Hg 6s")
    val_labels[i] = val_labels[i].replace("Cu1 1s", "Cu1 4s")
    val_labels[i] = val_labels[i].replace("Cu1 2p", "Cu1 4p")
    val_labels[i] = val_labels[i].replace("Cu2 1s", "Cu2 4s")
    val_labels[i] = val_labels[i].replace("Cu2 2p", "Cu2 4p")
pmol_val.ao_labels = lambda *args: val_labels
print ("Valence:")
print (len(pmol_val.ao_labels()))
print (pmol_val.ao_labels())

# CORE
minao_core = 'def2-svp-atom-iao-core'
pmol_core = pmol.copy()
pmol_core.basis = minao_core
pmol_core.build()
basis_core["Cu1"] = copy.deepcopy(pmol_core._basis["Cu1"])
basis_core["Cu2"] = copy.deepcopy(pmol_core._basis["Cu2"])
basis_core["O1"] = copy.deepcopy(pmol_core._basis["O1"])
basis_core["O2"] = copy.deepcopy(pmol_core._basis["O2"])
basis_core["Hg"] = copy.deepcopy(pmol_core._basis["Hg"])
basis_core["Ba"] = copy.deepcopy(pmol_core._basis["Ba"])

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
    C_ao_iao_virt = Lat.symmetrize_lo(C_ao_iao_virt)
    C_ao_iao_xcore = make_basis.tile_C_ao_iao(C_ao_iao_val, C_virt=C_ao_iao_virt, C_core=None)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    
    mo_coeff = np.asarray(kmf.mo_coeff)
    mo_occ = np.asarray(kmf.mo_occ)

    print ("check IAO")
    print ("orthonormal")
    print (check_orthonormal(C_ao_iao_core,  ovlp, tol=1e-9))
    print (check_orthonormal(C_ao_iao_val,  ovlp, tol=1e-9))
    print (check_orthonormal(C_ao_iao_virt,  ovlp, tol=1e-9))
    print ("val core orth")
    print (check_orthogonal(C_ao_iao_val, C_ao_iao_core, ovlp, tol=1e-9))
    print ("virt core orth")
    print (check_orthogonal(C_ao_iao_virt, C_ao_iao_core, ovlp, tol=1e-9))
    print ("val virt orth")
    print (check_orthogonal(C_ao_iao_val, C_ao_iao_virt, ovlp, tol=1e-9))
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
    
mo_coeff = np.asarray(kmf.mo_coeff)
mo_occ = np.asarray(kmf.mo_occ)

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

lo_labels = get_labels(cell, minao=None, full_virt=False, B2_labels=pmol_val.ao_labels(), \
        core_labels=pmol_core.ao_labels())[0]
idx_spec = get_idx_each(cell, labels=lo_labels, kind='atom')
idx_spec_val = get_idx_each(cell, labels=pmol_val.ao_labels(), kind='atom')
idx_orb = get_idx_each(cell, labels=lo_labels, kind='id atom nl')
idx_all = get_idx_each(cell, labels=lo_labels, kind='all')

C_ao_lo = C_ao_iao_xcore

NCORE = ncore
NLO = C_ao_lo.shape[-1]
nlo = NLO

cell_lo = cell.copy()
cell_lo.nelectron = cell_lo.nelectron - NCORE * 2
cell_lo.nao_nr = lambda *args: NLO
cell_lo.build()

Lat = lattice.Lattice(cell_lo, kmesh)
Lat_ion = lattice.Lattice(cell_lo, kmesh)

Cu_idx = list(idx_spec["Cu1"] + list(idx_spec["Cu2"]))
O1_idx = idx_spec["O1"]
O2_idx = idx_spec["O2"]
#Ca_idx = idx_spec["Ca"]
ion_idx = np.sort(list(idx_spec["Hg"]) + list(idx_spec["Ba"]), kind='mergesort')
Cu_O1_idx = np.sort(np.append(np.append(Cu_idx, O1_idx), O2_idx), kind='mergesort')

Cu_idx_val = list(idx_spec_val["Cu1"]) + list(idx_spec_val["Cu2"])
O1_idx_val = idx_spec_val["O1"]
O2_idx_val = idx_spec_val["O2"]

Hg_idx_val = idx_spec_val["Hg"]
Ba_idx_val = idx_spec_val["Ba"]

Cu_O1_idx_val = np.sort(np.append(np.append(Cu_idx_val, O1_idx_val), O2_idx_val), kind='mergesort')
Cu_O1_idx_virt = [idx for idx in Cu_O1_idx if not idx in Cu_O1_idx_val]

ion_idx_val = np.sort(np.append(Hg_idx_val, Ba_idx_val), kind='mergesort')
ion_idx_virt = [idx for idx in ion_idx if not idx in ion_idx_val]



print (idx_all.keys())

idx_sp_orb = get_idx_each(cell_lo, labels=lo_labels, kind='atom nl')
idx_all = get_idx_each(cell_lo, labels=lo_labels, kind='all')
idx_x = get_idx_each(cell, labels=lo_labels, kind='atom nlm')
O_3band =   list(idx_all["4 O1 2py   "])     + list(idx_all["5 O1 2px   "])\
          + list(idx_all["6 O1 2py   "])     + list(idx_all["7 O1 2px   "])\
          + list(idx_all["8 O1 2px   "])     + list(idx_all["9 O1 2py   "])\
          + list(idx_all["10 O1 2px   "])     + list(idx_all["11 O1 2py   "])\
          + list(idx_x["O2 2pz"])

act_idx = np.sort(list(idx_x["Cu1 3dx2-y2"]) \
                + list(idx_x["Cu2 3dx2-y2"]) \
                + O_3band, kind='mergesort')
tband_idx = act_idx

print ("Fitting labels")
print (np.asarray(lo_labels)[tband_idx])

print ("imp labels")
print (np.asarray(lo_labels)[Cu_O1_idx_val])
print (np.asarray(lo_labels)[Cu_O1_idx_virt])
print (np.asarray(lo_labels)[ion_idx])

Lat.set_val_virt_core(Cu_O1_idx_val, Cu_O1_idx_virt, [])
Lat_ion.set_val_virt_core(ion_idx_val, ion_idx_virt, [])

from libdmet import utils
imp_idx_fit = utils.search_idx1d(tband_idx, Lat.imp_idx)

import libdmet.dmet.Hubbard as dmet
from libdmet.system.hamiltonian import HamNonInt

Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
        rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=(kmf.energy_nuc()+E_core).real)
Lat_ion.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4, hcore=hcore_xcore, ovlp=ovlp, \
        rdm1=rdm1_xcore, vj=vj_xcore, vk=vk_xcore, H0=0.0)

# system
Filling = [(cell_lo.nelectron) / float(Lat.nscsites*2.0),
           (cell_lo.nelectron) / float(Lat.nscsites*2.0)]
restricted = False
bogoliubov = False
int_bath = True
nscsites = Lat.nscsites
Mu = 0.0
last_dmu = 0.0
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
diis_start = 6
dc = dmet.FDiisContext(adiis.space)
trace_start = 1000

# solver and mu fit
cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-6, tol_normt=2e-5, max_memory=cell.max_memory)
cisolver_ion = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-6, tol_normt=2e-5, max_memory=cell.max_memory)

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
solver_ion.name = 'dmrgci_ion'

nelec_tol = 1.0e-6
delta = 0.01
step = 1.0
load_frecord = False

# vcor fit
imp_fit = False
emb_fit_iter = 1000 # embedding fitting
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

idx_atom = get_idx_each(cell, labels=np.asarray(lo_labels)[Lat.imp_idx], kind='id atom')

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    
    log.section("\nSolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %s", Mu)
    
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    rho_k = res["rho_k"]
    
    Lat.mulliken_lo_R0(rho[:, 0], labels=lo_labels)
    
    # Hg Ba
    log.section("\nConstructing impurity problem\n")
    ImpHam_ion, H1e_ion, basis_ion = dmet.ConstructImpHam(Lat_ion, rho, vcor, matching=False, \
            int_bath=int_bath, max_memory=cell.max_memory, t_reversal_symm=True, \
            orth=True)
    ImpHam_ion = dmet.apply_dmu(Lat_ion, ImpHam_ion, basis_ion, last_dmu)
    basis_k_ion = Lat_ion.R2k_basis(basis_ion)

    log.section("\nSolving impurity problem\n")
    ci_args_ion = {"restart": True, "fcc_name": "fcc_ion.h5"}
    solver_args_ion = {"nelec": (Lat_ion.ncore+Lat_ion.nval)*2, \
            "guess": dmet.foldRho_k(rho_k, basis_k_ion), \
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
    ci_args = {"restart": True, "fcc_name": "fcc.h5"}
    solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
            "guess": dmet.foldRho_k(rho_k, basis_k), \
            "basis": basis,
            "ci_args": ci_args}
    
    rhoEmb_col, EnergyEmb_col, ImpHam_col, dmu = \
        dmet.SolveImpHam_with_fitting([Lat, Lat_ion], Filling, [ImpHam, ImpHam_ion], \
        [basis, basis_ion], [solver, solver_ion], \
        solver_args=[solver_args, solver_args_ion], thrnelec=nelec_tol, \
        delta=delta, step=step)
    ImpHam, ImpHam_ion = ImpHam_col
    rhoEmb, rhoEmb_ion = rhoEmb_col
    EnergyEmb, EnergyEmb_ion = EnergyEmb_col
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    
    basis_col = [basis, basis_ion]
    Lat_col = [Lat, Lat_ion]
    rho_glob = slater.get_rho_glob_k(basis_col, Lat_col, rhoEmb_col, compact=True)
    veff = slater.get_veff_from_rdm1_emb(Lat_col, rhoEmb_col, basis_col)
    
    np.save("basis.npy", basis)
    np.save("JK_core.npy", Lat.JK_core)

    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args, veff=veff)
    EnergyImp *= nscsites
    log.result("last_dmu = %20.12f", last_dmu)
    log.result("E(DMET) = %20.12f", EnergyImp)
    
    Lat.mulliken_lo_R0(rhoImp, labels=np.asarray(lo_labels)[Lat.imp_idx])
    
    rhoImp_ion, EnergyImp_ion, nelecImp_ion = \
            dmet.transformResults(rhoEmb_ion, EnergyEmb_ion, basis_ion, ImpHam_ion, H1e_ion, \
            lattice=Lat_ion, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver_ion, solver_args=solver_args_ion, veff=veff)
    EnergyImp_ion *= nscsites
    log.result("E(DMET ion) = %20.12f", EnergyImp_ion)
    
    EnergyImp += EnergyImp_ion
    print ("Energy (DMET) = %s"%EnergyImp)

    # DUMP results:
    dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
            Lat.getFock(kspace=False), rho], dtype=object)
    np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)
    
    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter, method='CG', \
            imp_fit=imp_fit, ytol=ytol, gtol=gtol, CG_check=CG_check, imp_idx=imp_idx_fit)

    if iter >= trace_start:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)

    dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
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


