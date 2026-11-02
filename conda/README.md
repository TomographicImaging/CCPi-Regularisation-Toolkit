# ccpi-regulariser conda package

## prerequisites

```sh
conda install conda-forge::rattler-build
git clone https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit
cd CCPi-Regularisation-Toolkit
```

## rattler-build (with ccache)

- Platforms: `linux-64`, `win-64`
- CUDA: `None` (CPU), `12`, `13`

See the top of [recipe.yaml](./recipe.yaml) for invocation.

## install

```sh
rattler-build publish output/*64/*.conda --to ./channel
# install in env
conda create -n ccpi python -c file://$PWD/channel -c conda-forge -c ccpi ccpi-regulariser
conda activate ccpi
```

## Notes

- `-c channel` order matters
- `-v` for verbose build output
- disable ccache:
  ```sh
  CMAKE_C_COMPILER_LAUNCHER="" CMAKE_CXX_COMPILER_LAUNCHER="" CMAKE_CUDA_COMPILER_LAUNCHER="" rattler-build ...
  ```
