#include <ATen/ATen.h>

#include <float.h>

#include <stdio.h>

#define THREADS (512 + 128)
#define BLOCKS 28

#define FDIM 64
#define PDIM 2

__global__ void
spatially_aware_nn_kernel(const float *__restrict__ x, const float *__restrict__ y,
                          const float *__restrict__ pos_x, const float *__restrict__ pos_y,
                          float pos_dist_threshold,
                          int64_t *__restrict__ nn_idx,
                          size_t num_x,
                          size_t num_y) {
  const ptrdiff_t block_idx = blockIdx.x;
  const ptrdiff_t thread_idx = threadIdx.x;
  const ptrdiff_t start_idx_x = block_idx * THREADS + thread_idx;

  for (ptrdiff_t n_x = start_idx_x; n_x < num_x; n_x += BLOCKS * THREADS) {
    float pos_x_c[PDIM];
    #pragma unroll
    for (ptrdiff_t d = 0; d < PDIM; ++d) {
      pos_x_c[d] = pos_x[n_x * PDIM + d];
    }

    float x_c[FDIM];
    #pragma unroll
    for (ptrdiff_t d = 0; d < FDIM; ++d) {
      x_c[d] = x[n_x * FDIM + d];
    }

    float sim = -1 - FLT_EPSILON;
    ptrdiff_t best_n_y = -1;

    for (ptrdiff_t n_y = 0; n_y < num_y; n_y++) {
      float tmp_pos_dist = 0;
      #pragma unroll
      for (ptrdiff_t d = 0; d < PDIM; d++) {
        tmp_pos_dist = fmaxf(tmp_pos_dist, fabsf(pos_x_c[d] - pos_y[n_y * PDIM + d]));
      }
      
      float tmp_sim = 0;
      #pragma unroll
      for (ptrdiff_t d = 0; d < FDIM; d++) {
        tmp_sim += x_c[d] * y[n_y * FDIM + d];
      }
      
      if (sim < tmp_sim && tmp_pos_dist > pos_dist_threshold) {
        sim = tmp_sim;
        best_n_y = n_y;
      }
    }

    nn_idx[n_x] = best_n_y;
  }
}

at::Tensor spatially_aware_nn_cuda(at::Tensor x, at::Tensor y, 
                                   at::Tensor pos_x, at::Tensor pos_y,
                                   float pos_dist_threshold) {
  at::TensorOptions options = x.options().dtype(at::ScalarType::Long);
  auto nn_idx = at::full(x.size(0), -1, options);

  spatially_aware_nn_kernel<<<BLOCKS, THREADS>>>(
        x.data<float>(), y.data<float>(), 
        pos_x.data<float>(), pos_y.data<float>(), 
        pos_dist_threshold,
        nn_idx.data<int64_t>(),
        x.size(0),
        y.size(0)
  );

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("%s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("%s\n", cudaGetErrorString(errAsync));

  return nn_idx;
}
