# Spatially Aware Nearest Neighbors for PyTorch

Nearest neighbors are already "spatially aware" in the feature space. This package also considers a second space called "the position space" - which is often the case for local features. The current use-case is negative mining, but the code can be adapted for other tasks such as guided matching. 

Please note that the grid parameters were optimized for one GTX 1080Ti. For the moment, the feature and position dimensionalities are hardcoded in the CUDA code.

## Running tests

`python setup.py test`
