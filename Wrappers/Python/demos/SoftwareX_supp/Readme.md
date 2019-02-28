
# SoftwareX publication [1] supporting files

## Decription:
The scripts here support publication in SoftwareX journal [1] to ensure reproducibility of the research. The scripts linked with data shared at Zenodo. 

## Data:
Data is shared at Zenodo [here](https://doi.org/10.5281/zenodo.2578893)

## Dependencies:
1. [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox): `conda install -c astra-toolbox astra-toolbox`
2. [TomoRec](https://github.com/dkazanc/TomoRec): `conda install -c dkazanc tomorec`
3. [Tomophantom](https://github.com/dkazanc/TomoPhantom): `conda install tomophantom -c ccpi`

## Files description: 
- `Demo_SimulData_SX.py` - simulates 3D projection data using [Tomophantom](https://github.com/dkazanc/TomoPhantom) software. One can skip this module if the data is taken from [Zenodo](https://doi.org/10.5281/zenodo.2578893)
- `Demo_SimulData_ParOptimis_SX.py` - runs computationally extensive calculations for optimal regularisation parameters, the result are saved into directory `optim_param`. This script can be also skipped. 
- `Demo_SimulData_Recon_SX.py` - using established regularisation parameters, one runs iterative reconstruction
- `Demo_RealData_Recon_SX.py` - runs real data reconstructions. Can be quite intense on memory so reduce the size of the reconstructed volume if needed. 

### References:
[1] "CCPi-Regularisation Toolkit for computed tomographic image reconstruction with proximal splitting algorithms" by Daniil Kazantsev, Edoardo Pasca, Martin J. Turner and Philip J. Withers; SoftwareX, 2019. 

### Acknowledgments:
CCPi-RGL software is a product of the [CCPi](https://www.ccpi.ac.uk/) group, STFC SCD software developers and Diamond Light Source (DLS). Any relevant questions/comments can be e-mailed to Daniil Kazantsev at dkazanc@hotmail.com

