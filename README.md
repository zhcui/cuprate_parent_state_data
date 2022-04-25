# Cuprate Parent State Data

Data repository for paper https://arxiv.org/abs/2112.09735

Systematic electronic structure in the cuprate parent state from quantum many-body simulations

Prerequisites
-------------

To fully reproduce the results, please use the following codes:

* [PySCF](https://github.com/zhcui/pyscf/tree/kpoint_symm) 
	- make sure you use the branch `kpoint_symm` and the linked version of pyscf.
	- including the basis for calculation, IAO references, modification in initial guess, density fitting and allow kpoints symmetry.

* [libDMET](https://github.com/gkclab/libdmet_preview)
	- main DMET library

* [block2](https://github.com/block-hczhai/block2-preview)
    - DMRG solver

* [VASP](https://vasp.at)
    - basis completeness check with plane wave basis

* [SpinW](https://spinw.org)
    - spin-wave calculations

Directory
---------

* 01_crystal_geometry: crystal structures in VASP-POSCAR format.
	- Hg-1201
	- Hg-1212
	- CCO
	- CuO2
	- La2CuO4

* 02_benchmark:
	- 02_data.xlsx
	- example_scripts

* 03_dmet: DMET for Hg-Ba-Ca-Cu-O systems.
	- 03_data.xlsx
	- Hg-1201
	- Hg-1212
	- CCO
	- CuO2

* 04_other_script:
	- iao_reference: script for generating reference local orbitals.
