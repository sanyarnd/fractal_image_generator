#include "renderer_cuda.hpp"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#include "fractal_data.hpp"
#include "rand_gen_thurst.hpp"
#include "transforms.hpp"
#include "utils.hpp"

namespace fractal {

void check_cuda_info(bool extended = false);

__global__ void render(pix *pixels,
                       uint2 size,
                       float4 view,
                       coeff *fc,
                       short fc_size,
                       unsigned short *transformations,
                       short transformations_size,
                       float rot_angle,
                       unsigned int seed,
                       short iter_per_sample);

image render_cuda(unsigned int width,
                  unsigned int height,
                  flame_coeffs const &fc,
                  unsigned long int seed,
                  int samples,
                  short iter_per_sample,
                  std::vector<unsigned short> const &trans,
                  rect const &view,
                  short symmetry) {
  using namespace std::chrono;
  std::cout << "CUDA rendering, gathering device information...\n";
  check_cuda_info();

  // timing measure
  auto start = system_clock::now();
  cudaEvent_t start_kernel, stop_kernel;
  cudaCheckRet(cudaEventCreate(&start_kernel));
  cudaCheckRet(cudaEventCreate(&stop_kernel));

  unsigned int pix_cnt = width * height;
  short fc_siz = static_cast<short>(fc.size());
  short tr_sz = static_cast<short>(trans.size());

  // buffers
  coeff *d_fc;
  pix *d_pixels;
  unsigned short *d_trans;
  cudaCheckRet(cudaMalloc(&d_fc, fc_siz * sizeof(coeff)));
  cudaCheckRet(cudaMalloc(&d_pixels, pix_cnt * sizeof(pix)));
  cudaCheckRet(cudaMalloc(&d_trans, tr_sz * sizeof(unsigned short)));

  // populate
  std::cout << "Copying coeffs to GPU\n";
  cudaCheckRet(cudaMemcpy(d_fc, fc.data(), fc_siz * sizeof(coeff), cudaMemcpyHostToDevice));
//  std::cout << "Initializing buffer with zeros...\n";
//  cudaCheckRet(cudaMemset(d_pixels, 0, pix_cnt * sizeof(pix)));
  std::cout << "Copying misc data...\n";
  cudaCheckRet(cudaMemcpy(d_trans, trans.data(), tr_sz * sizeof(unsigned short), cudaMemcpyHostToDevice));

  std::cout << "Running kernel...\n";
  cudaEventRecord(start_kernel);
  render << < samples, 1 >> > (d_pixels,
      {width, height},
      {view.x, view.y, view.x_end(), view.y_end()},
      d_fc, fc_siz, d_trans, tr_sz, pi_two() / symmetry, seed, iter_per_sample);
  cudaCheckError();
  cudaDeviceSynchronize();
  cudaEventRecord(stop_kernel);
  cudaEventSynchronize(stop_kernel);
  std::cout << "kernel finished!\n";

  // kernel time
  float umTime = 0;
  cudaEventElapsedTime(&umTime, start_kernel, stop_kernel);

  // copy the resulting image
  std::cout << "copying data from device to host...\n";
  auto pixels = std::make_unique<pix[]>(pix_cnt);
  cudaCheckRet(cudaMemcpy(pixels.get(), d_pixels, pix_cnt * sizeof(pix), cudaMemcpyDeviceToHost));
  image result_image(width, height, pixels.get());

  // cleanup
  std::cout << "cleaning up data...\n";
  cudaEventDestroy(start_kernel);
  cudaEventDestroy(stop_kernel);
  cudaFree(d_pixels);
  cudaFree(d_fc);
  cudaFree(d_trans);

  // total time
  auto end = system_clock::now();
  auto elapsed = duration_cast<milliseconds>(end - start);
  std::cout << "rendering completed!\n";
  std::cout << "render time: " << elapsed.count() << "ms; kernel time " << std::to_string(umTime) << "ms\n\n";

  return result_image;
}

__global__ void render(pix *pixels,
                       uint2 size,
                       float4 view,
                       coeff *fc,
                       short fc_size,
                       unsigned short *transformations,
                       short transformations_size,
                       float rot_angle,
                       unsigned int seed,
                       short iter_per_sample) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  rand_gen_thurst rg(seed * idx);

  float new_x = rg.rand_float(view.x, view.z);
  float new_y = rg.rand_float(view.y, view.w);

  for (short step = 0; step < iter_per_sample; ++step) {
    short ci = static_cast<short>(rg.rand_uint(0, std::numeric_limits<int>::max()) % fc_size);

    // on each step we take the next transformation from the list
    unsigned short transform = transformations[idx % transformations_size];
    // get new new_x/new_y
    transform_xy(&new_x, &new_y, transform, fc + ci, rg.rand_float(), rg.rand_float(), rg.random_bool());

    for (float theta2 = 0.0; theta2 < pi_two(); theta2 += rot_angle) {
      float x_rot = new_x * cosf(theta2) - new_y * sinf(theta2);
      float y_rot = new_x * sinf(theta2) + new_y * cosf(theta2);

      if (x_rot >= view.x && x_rot <= view.z && y_rot >= view.y && y_rot <= view.w) {
        auto x1 = static_cast<unsigned int>(map_range(view.x, view.z, 0, size.x, x_rot));
        auto y1 = static_cast<unsigned int>(map_range(view.y, view.w, 0, size.y, y_rot));

        // store successful values per block
        if (image::contains(size.x, size.y, x1, y1)) {
          pix *p = mat_offset_elem(pixels, y1, x1, size.x);

          unsigned char r = fc[ci].r;
          unsigned char g = fc[ci].g;
          unsigned char b = fc[ci].b;
          // add base color on the first iteration
          if (p->inc_cnt == 0) {
            p->r = r;
            p->g = g;
            p->b = b;
          } else {
            p->r = static_cast<unsigned char>((p->r + r) / 2);
            p->g = static_cast<unsigned char>((p->g + g) / 2);
            p->b = static_cast<unsigned char>((p->b + b) / 2);
          }
          atomicAdd(&p->inc_cnt, 1);
        }
      }
    }
  }
}

void check_cuda_info(bool extended) {
  int deviceCount;
  cudaCheckRet(cudaGetDeviceCount(&deviceCount));
  std::cout << "Number of CUDA devices " << deviceCount << "\n";
  if (deviceCount == 0) {
    std::cout << "CUDA device not found, exiting...\n";
    exit(EXIT_FAILURE);
  }
  cudaDeviceProp prop;
  const int dev = 0;
  cudaCheckRet(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Using `" << prop.name << "`\n";

  std::cout << "Total Global Memory:        " << (prop.totalGlobalMem / (1024 * 1024)) << "MB\n";
  std::cout << "Total shared mem per block: " << (prop.sharedMemPerBlock / 1024) << "kb\n";
  std::cout << "Total const mem size:       " << (prop.totalConstMem / 1024) << "kb\n";
  std::cout << "Warp size:                  " << prop.warpSize << "\n";
  if (extended) {
    std::cout << "Maximum block dimensions:   " << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "Maximum grid dimensions:    " << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << "\n";
    std::cout << "Clock Rate:                 " << prop.clockRate << "\n";
    std::cout << "Number of muliprocessors:   " << prop.multiProcessorCount << "\n";
  }
  std::cout << "\n";
}

}

