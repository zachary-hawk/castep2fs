# castep2fs


A CASTEP utility for taking a CASTEP output and producing publication quality Fermi Surfaces and related quantities.

Requirements
------------

This code is built upon the wonderful PyVista platform (https://docs.pyvista.org) and makes use of the Atomic Simulation Environment (https://wiki.fysik.dtu.dk/ase).

Usage
-----

castep2fs reads castep input and output files.
```
input : <seed>.cell
output: <seed>.bands
        <seed>-out.cell (optional but improves performance)



castep2fs <seed>
```

For help run:
```
castep2fs -h
```