#!/usr/bin/env python

"""
Generate IAO reference from atomic HF.
"""

import copy
import numpy as np

from pyscf.data import elements
from pyscf.lib import logger
from pyscf.scf.atom_hf import AtomSphAverageRHF
from pyscf.gto.basis import parse_nwchem
from pyscf.gto.basis.parse_nwchem import SPDF
from pyscf.data.elements import _std_symbol

def convert_basis_to_nwchem(symb, basis):
    '''Convert the internal basis format to NWChem format string'''
    res = []
    symb = _std_symbol(symb)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('#BASIS SET: (%s) -> [%s]' % (','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    for bas in basis:
        res.append('%-2s    %s' % (symb, SPDF[bas[0]]))
        for dat in bas[1:]:
            res.append(' '.join('%27.17f'%x for x in dat))
    return '\n'.join(res)

def get_atomic_hf_basis(mol, atomic_configuration=elements.NRSRHF_CONFIGURATION, 
                        charge_dic=None, tol=1e-7):
    """
    Get atomic HF basis set.

    Returns:
        string: NWChem format basis set for every element in mol.
    """
    string = ""
    ele_dic = set()
    
    elements = set([a[0] for a in mol._atom])
    logger.info(mol, 'Spherically averaged atomic HF for %s', elements)

    atm_template = copy.copy(mol)
    atm_template.charge = 0
    atm_template.symmetry = False  # TODO: enable SO3 symmetry here
    atm_template.atom = atm_template._atom = []
    atm_template.cart = False  # AtomSphAverageRHF does not support cartensian basis

    atm_scf_result = {}
    for ia, a in enumerate(mol._atom):
        element = a[0]
        if element in atm_scf_result:
            continue

        atm = atm_template
        atm._atom = [a]
        atm._atm = mol._atm[ia:ia+1]
        atm._bas = mol._bas[mol._bas[:,0] == ia].copy()
        atm._ecpbas = mol._ecpbas[mol._ecpbas[:,0] == ia].copy()
        # Point to the only atom
        atm._bas[:,0] = 0
        atm._ecpbas[:,0] = 0
        if element in mol._pseudo:
            atm._pseudo = {element: mol._pseudo.get(element)}
            raise NotImplementedError
        atm.spin = atm.nelectron % 2

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            mo_occ = mo_energy = numpy.zeros(nao)
            mo_coeff = numpy.zeros((nao,nao))
            atm_scf_result[element] = (0, mo_energy, mo_coeff, mo_occ)
        else:
            if atm.nelectron == 1:
                atm_hf = AtomHF1e(atm)
            else:
                atm_hf = AtomSphAverageRHF(atm)
                atm_hf.atomic_configuration = atomic_configuration

            atm_hf.verbose = mol.verbose
            atm_hf.run()
            atm_scf_result[element] = (atm_hf.e_tot, atm_hf.mo_energy,
                                       atm_hf.mo_coeff, atm_hf.mo_occ)

        labels = atm.ao_labels()
        labels_2 = atm.ao_labels(fmt=False)
        mo_coeff = atm_hf.mo_coeff
        bas = atm._basis[element]

        idx = []
        exists = set()
        for i, lab in enumerate(labels_2):
            if lab[2] in exists:
                continue
            else:
                idx.append(i)
                exists.add(lab[2])
        
        basis = []
        for j in idx:
            print (j, labels[j])
            new_bas = []
            mo = mo_coeff[idx, j]
            for i, bas_cur in enumerate(bas):
                l = bas_cur[0]
                exp_list = bas_cur[1:]
                c_list = atm.bas_ctr_coeff(i)
                assert c_list.shape[-1] == 1
                c_list = c_list.ravel()
                
                for (exp, _), coeff in zip(exp_list, c_list):
                    if abs(coeff * mo[i]) > tol:
                        new_bas.append([l, [exp, coeff * mo[i]]])
            l_list, tmp = list(zip(*new_bas))
            assert len(np.unique(l_list)) == 1
            basis.append([l_list[0], *tmp])
        
        basis = parse_nwchem.to_general_contraction(basis)

        if not _std_symbol(element) in ele_dic:
            string += convert_basis_to_nwchem(element, basis)
            string += "\n"
            ele_dic.add(_std_symbol(element))
    return string

if __name__ == "__main__":
    from libdmet_solid.utils import read_poscar
    
    np.set_printoptions(4, suppress=True, linewidth=1000)

    cell = read_poscar(fname="./Hg-1212-AFM.vasp")
    cell.basis = 'def2-svp-no-diffuse'
    cell.ecp = {"Hg": 'def2-svp', "Ba": "def2-svp"}
    cell.spin = 0 
    cell.verbose = 5
    cell.max_memory = 119000
    cell.precision = 1e-12
    cell.build()

    string = get_atomic_hf_basis(cell, tol=1e-7)
    fname = "def2-svp-atom.dat"
    with open(fname, 'w') as f:
        f.write(string)
