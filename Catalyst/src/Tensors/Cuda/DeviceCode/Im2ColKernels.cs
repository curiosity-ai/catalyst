//using ManagedCuda;
//using ManagedCuda.BasicTypes;
//using ManagedCuda.VectorTypes;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;

//namespace Catalyst.Tensors.CUDA.DeviceCode
//{
//    [Precompile]
//    public class Im2ColKernels : CudaCode
//    {

//        public static readonly string Code = @"
//#define CUDA_KERNEL_LOOP(i, n) \
//  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

//// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
//template <typename Dtype>
//__device__ void im2col_kernel_t(const int n, const Dtype* data_im,
//                              const int height, const int width,
//                              const int ksize_h, const int ksize_w,
//                              const int pad_h, const int pad_w,
//                              const int stride_h, const int stride_w,
//                              const int dilation_h, const int dilation_w,
//                              const int height_col, const int width_col,
//    Dtype* data_col) {
//  CUDA_KERNEL_LOOP(index, n) {
//    int w_out = index % width_col;
//    index /= width_col;
//    int h_out = index % height_col;
//    int channel_in = index / height_col;
//    int channel_out = channel_in * ksize_h * ksize_w;
//    int h_in = h_out * stride_h - pad_h;
//    int w_in = w_out * stride_w - pad_w;
//    data_col += (channel_out * height_col + h_out) * width_col + w_out;
//    data_im += (channel_in * height + h_in) * width + w_in;
//    for (int i = 0; i < ksize_h; ++i) {
//      for (int j = 0; j < ksize_w; ++j) {
//        int h = h_in + i * dilation_h;
//        int w = w_in + j * dilation_w;
//        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
//          data_im[i * dilation_h * width + j * dilation_w] : 0;
//        data_col += height_col * width_col;
//      }
//    }
//  }
//}

//template <typename Dtype>
//__device__ void col2im_kernel_t(const int n, const Dtype* data_col,
//                                  const int height, const int width, const int channels,
//                                  const int kernel_h, const int kernel_w,
//                                  const int pad_h, const int pad_w,
//                                  const int stride_h, const int stride_w,
//                                  const int dilation_h, const int dilation_w,
//                                  const int height_col, const int width_col,
//                                  Dtype* data_im) {
//  CUDA_KERNEL_LOOP(index, n) {
//    Dtype val = 0;
//    const int w_im = index % width + pad_w;
//    const int h_im = (index / width) % height + pad_h;
//    const int c_im = index / (width * height);
//    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
//    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
//    // compute the start and end of the output
//    const int w_col_start =
//      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
//    const int w_col_end = min(w_im / stride_w + 1, width_col);
//    const int h_col_start =
//      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
//    const int h_col_end = min(h_im / stride_h + 1, height_col);
//    // TODO: use LCM of stride and dilation to avoid unnecessary loops
//    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
//      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
//        int h_k = (h_im - h_col * stride_h);
//        int w_k = (w_im - w_col * stride_w);
//        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
//          h_k /= dilation_h;
//          w_k /= dilation_w;
//          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
//                                height_col + h_col) * width_col + w_col;
//          val += data_col[data_col_index];
//        }
//      }
//    }
//    data_im[index] = val;
//  }
//}

//extern ""C"" {
//    __global__ void im2col_kernel(const int n, const float* data_im,
//                              const int height, const int width,
//                              const int ksize_h, const int ksize_w,
//                              const int pad_h, const int pad_w,
//                              const int stride_h, const int stride_w,
//                              const int dilation_h, const int dilation_w,
//                              const int height_col, const int width_col, float* data_col)
//    {
//        im2col_kernel_t(n, data_im, height, width, ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);
//    }

//    __global__ void col2im_kernel(const int n, const float* data_col,
//                                  const int height, const int width, const int channels,
//                                  const int kernel_h, const int kernel_w,
//                                  const int pad_h, const int pad_w,
//                                  const int stride_h, const int stride_w,
//                                  const int dilation_h, const int dilation_w,
//                                  const int height_col, const int width_col,
//                                  float* data_im)
//    {
//        col2im_kernel_t(n, data_col, height, width, channels, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_im);
//    }
//}

//";

//        public Im2ColKernels() : base(Code)
//        {
//        }


//        public void Im2Col(Tensor im, Tensor col, int channels,
//            int height, int width,
//            int ksize_h, int ksize_w, int pad_h,
//            int pad_w, int stride_h, int stride_w,
//            int dilation_h, int dilation_w)
//        {
//            var context = CudaHelpers.TSContextForTensor(im);
//            var cudaContext = context.CudaContextForTensor(im);

//            // From Torch source:
//            // We are going to launch channels * height_col * width_col kernels, each
//            // kernel responsible for copying a single-channel grid.
//            int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1))
//                             / stride_h + 1;
//            int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1))
//                            / stride_w + 1;
//            int num_kernels = channels * height_col * width_col;

//            var data_im = CudaHelpers.GetBufferStart(im);
//            var data_col = CudaHelpers.GetBufferStart(col);

//            Invoke(context, cudaContext, "im2col_kernel", new dim3(NNThreads.NumBlocks(num_kernels)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
//                num_kernels, data_im, height, width, ksize_h, ksize_w,
//                pad_h, pad_w, stride_h, stride_w,
//                dilation_h, dilation_w,
//                height_col, width_col, data_col);
//        }

//        public void Col2Im(Tensor col, Tensor im, int channels, int height, int width,
//            int patch_h, int patch_w, int pad_h,
//            int pad_w, int stride_h, int stride_w,
//            int dilation_h, int dilation_w)
//        {
//            var context = CudaHelpers.TSContextForTensor(im);
//            var cudaContext = context.CudaContextForTensor(im);


//            int height_col = (height + 2 * pad_h - (dilation_h * (patch_h - 1) + 1))
//                   / stride_h + 1;
//            int width_col = (width + 2 * pad_w - (dilation_w * (patch_w - 1) + 1))
//                             / stride_w + 1;
//            int num_kernels = channels * height * width;

//            var data_im = CudaHelpers.GetBufferStart(im);
//            var data_col = CudaHelpers.GetBufferStart(col);

//            // From Torch source:
//            // To avoid involving atomic operations, we will launch one kernel per
//            // bottom dimension, and then in the kernel add up the top dimensions.

//            Invoke(context, cudaContext, "col2im_kernel", new dim3(NNThreads.NumBlocks(num_kernels)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
//                num_kernels, data_col, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
//                dilation_h, dilation_w,
//                height_col, width_col, data_im);
//        }

//        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
//        {
//            var ptx = GetPtx(context.Compiler);
//            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
//            kernel.GridDimensions = grid;
//            kernel.BlockDimensions = block;
//            kernel.DynamicSharedMemory = smemSize;
//            kernel.RunAsync(stream, args);
//        }


//    }
//}
