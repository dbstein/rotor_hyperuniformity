# rotor_hyperuniformity

Archive of code used to generate results in "Hyperuniformity and phase enrichment in vortex and rotor assemblies", currently under review. This readme will be updated with publication information once the paper is in print.

## Contents
1. *simulate.py*: The main driver for simulations.
2. *kernels.py*: Fast, numba-jitted direct kernels for timestepping Euler, QSG, and general Saffman-Delbr&uuml;ck vortices/rotors, along with a few convenience functions.
3. *generate_initial_data.py*: Initial data generation via rejection sampling.
4. *saffman.py*: Machinery to generate fast, jit-compatible functions for evaluating the general Saffman-Delbr&uuml;ck Kernels. Depends on *function_generator.py* (see below).
5. *analyze.py*: Functions to compute the structure factor and radially averaged structure factor. Depends on installation of FINUFFT (see requirements).
6. *function_generator.py*: A frozen version of the function_generator.py tool, hosted at [https://github.com/dbstein/function_generator](https://github.com/dbstein/function_generator). This version has been slightly modified in comparison to the github version to allow its direct inclusion in this package in a static manner. This introduces the additional dependence of *packaging* (tested with version 20.9), which can be installed via standard package managers (pip or conda). This code will not be imported unless the user requests to use the general Saffman-Delbr&uuml;ck kernels.

## Requriements
1. Simulation of the Euler and QSG kernels needs only a basic installation of python with *numpy*, *scipy*, and *numba*. The code in this repository has been tested using python version 3.9.4, numpy version 1.20.1, scipy version 1.6.2, and numba version 0.53.1.
2. Simulation of the general Saffman-Delbr&uuml;ck kernels additionally requires importing *function_generator.py* (included) and dependencies, see above.
3. The function to compute the structure factor additionally requires the installation of the Flatiron Institute Nonuniform Fast Fourier Transform, or FINUFFT, which can be downloaded at [https://github.com/flatironinstitute/finufft](https://github.com/flatironinstitute/finufft).
4. The final plots produced by the analysis code require *matplotlib* (tested using version 3.3.4).
5. This code uses direct summation to evaluate the many-body interaction problem, and exploits parallelized kernels to reduce the computation time. We ran this code on 128-core compute nodes with 2 AMD Rome Epyc processors; on these machines this code was somewhat faster than using O(N) algorithms (FMMs) for 10,000 rotors or vortices. On smaller machines the code may be quite slow. The example problem we have setup in *simulate.py* thus uses only 1,000 rotors/vortices rather than the 10,000 rotors/vortices used in most simulations in the paper.

## Instructions for use

The file *simulate.py* allows the simulation of arbitrary size ensembles of different initial radii, using either Euler, QSG, or the general Greens function with abritrary Saffman-Delbr&uuml;ck number, with or without steric interactions. Half of the particles can be configured to have a different circulation than the other half. Parameters are set directly in the file.

Simulations are started with random initial data and integrated from t=0.0 to a specified final time using [sp.integrate.DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html) (an eigth-order adaptive Runge-Kutta method).  Basic analysis is done on the final result, generating plots of the structure factor and the radially averaged structure factor, along with the trajectory of a single particle and the relative error in the Hamiltonian (which should only be conserved when steric repulsion is not present).

While the code can simply be invoked from the command line (e.g. as "python simulate.py"), running the code in IPython allows for interactively exploring the final data and plots. This code was tested in IPython version 7.22.0.

On a 4-core Macbook pro, the example configuration in *simulate.py* takes ~12 minutes to complete. Because this code makes heavy use of jit-compiled numba functions, there is substantial startup time, and it may take anywhere from several seconds to close to a minute for the first timestep to execute. All additional timesteps should be fast.
