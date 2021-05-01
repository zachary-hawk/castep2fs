from setuptools import find_packages
from setuptools import setup,Extension

setup(name="castep2fs",
      version="1.0.0",
      packages=find_packages(),
      description="CASTEP utility for calculating Fermi surfaces.",
      url="https://github.com/zachary-hawk/castep2fs.git",
      author="Zachary Hawkhead",
      author_email="zachary.hawkhead@durham.ac.uk",
      license="MIT",
      install_requires=["numpy",
                        "matplotlib",
                        "scipy",
                        "ase",
                        "pyvista",
                        "vtk",
                        "argparse","spglib"],

      entry_points={"console_scripts":["castep2fs=Source.main:main",]
                    }

      )

