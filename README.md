# Spatially Aware Nearest Neighbors for PyTorch

Nearest neighbors are already "spatially aware" in the feature space. This package also considers a second space called "the position space" - which is often the case for local features. The current use-case is negative mining, but the code can be adapted for other tasks such as guided matching. 

For the moment, the feature and position dimensionalities are hardcoded in the CUDA code.

# Releases

`v0.0.1`: The grid parameters are now compatible with most recent GPUs. The performance difference is marginal.

`v0.0.0`: Initial release.

## Running tests

`python setup.py test`
