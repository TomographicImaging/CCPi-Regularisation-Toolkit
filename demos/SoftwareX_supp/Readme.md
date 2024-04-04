# Software X journal publication supporting files

## Description:
The scripts support [publication](./paper/1-s2.0-S2352711018301912-main.pdf) in Software X journal [1] to ensure reproducibility of the research. The scripts linked with the data which is shared at [Zenodo](https://doi.org/10.5281/zenodo.2578893).

## Data:
Data is shared at Zenodo [here](https://doi.org/10.5281/zenodo.2578893)

## Dependencies:
1. [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox): `conda install -c astra-toolbox astra-toolbox`
2. [ToMoBAR](https://github.com/dkazanc/ToMoBAR): `conda install -c dkazanc tomobar`
3. [Tomophantom](https://github.com/dkazanc/TomoPhantom): `conda install tomophantom -c ccpi`

## Files description:
- `Demo_SimulData_SX.py` - simulates 3D projection data using [Tomophantom](https://github.com/dkazanc/TomoPhantom) software. One can skip this module if the data is taken from [Zenodo](https://doi.org/10.5281/zenodo.2578893)
- `Demo_SimulData_ParOptimis_SX.py` - runs computationally extensive calculations for optimal regularisation parameters, the result are saved into directory `optim_param`. This script can be also skipped.
- `Demo_SimulData_Recon_SX.py` - using established regularisation parameters, one runs iterative reconstruction
- `Demo_RealData_Recon_SX.py` - runs real data reconstructions. Can be quite intense on memory so reduce the size of the reconstructed volume if needed.

### References:
[1] [Kazantsev, D., Pasca, E., Turner, M.J. and Withers, P.J., 2019. CCPi-Regularisation toolkit for computed tomographic image reconstruction with proximal splitting algorithms. SoftwareX, 9, pp.317-323.](https://www.sciencedirect.com/science/article/pii/S2352711018301912)

### Acknowledgments:
CCPi-RGL software is a product of the [CCPi](https://www.ccpi.ac.uk/) group, STFC SCD software developers and Diamond Light Source (DLS). Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com
