#include <torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

at::Tensor spatially_aware_nn_cuda(at::Tensor x, at::Tensor y, 
                                   at::Tensor pos_x, at::Tensor pos_y, 
                                   float pos_dist_threshold);

at::Tensor spatially_aware_nn(at::Tensor x, at::Tensor y,
                              at::Tensor pos_x, at::Tensor pos_y, 
                              float pos_dist_threshold) {
  CHECK_CUDA(x);
  IS_CONTIGUOUS(x);
  CHECK_CUDA(y);
  IS_CONTIGUOUS(y);
  CHECK_CUDA(pos_x);
  IS_CONTIGUOUS(pos_x);
  CHECK_CUDA(pos_y);
  IS_CONTIGUOUS(pos_y);
  return spatially_aware_nn_cuda(x, y, pos_x, pos_y, pos_dist_threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spatially_aware_nn", &spatially_aware_nn, "Spatially Aware Nearest Neighbor (CUDA)");
}
