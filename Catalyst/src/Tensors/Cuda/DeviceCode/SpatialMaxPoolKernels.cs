using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Cpu;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class SpatialMaxPoolKernels : CudaCode
    {
        public static readonly string Code = @"

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// kernels borrowed from Caffe
template <typename Dtype>
__global__ void MaxPoolForward_t(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, Dtype* top_data,
    Dtype* top_mask, Dtype minVal) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height);
    int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = minVal;
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    top_mask[index] = maxidx + 1;
  }
}


template <typename Dtype>
__global__ void MaxPoolBackward_t(const int nthreads, const Dtype* top_diff,
    const Dtype* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
	if (top_mask[ph * pooled_width + pw] - 1 == h * width + w) {
	  gradient += top_diff[ph * pooled_width + pw];
	}
      }
    }
    bottom_diff[index] = gradient;
  }
}


extern ""C"" {

#define FLT_MIN 1.17549435e-38F

__global__ void MaxPoolForward(const int nthreads, const float* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, float* top_data,
    float* top_mask)
    {
        MaxPoolForward_t(nthreads, bottom_data, num, channels, height, width, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, top_data, top_mask, FLT_MIN);
    }

__global__ void MaxPoolBackward(const int nthreads, const float* top_diff,
    const float* top_mask, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* bottom_diff)
    {
        MaxPoolBackward_t(nthreads, top_diff, top_mask, num, channels, height, width, pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, bottom_diff);
    }

}

";

        public SpatialMaxPoolKernels() : base(Code)
        {
        }

        public void SpatialMaxPoolingForward(Tensor input, Tensor output, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var context = CudaHelpers.TSContextForTensor(input);
            var cudaContext = context.CudaContextForTensor(input);

            var iwidth = input.Sizes[3];
            var iheight = input.Sizes[2];
            var nInputPlane = input.Sizes[1];
            var batchSize = input.Sizes[0];

            long owidth;
            long oheight;

            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            if (cd.padW != 0 || cd.padH != 0)
            {
                // ensure that the last pooling starts inside the image
                if ((oheight - 1) * cd.dH >= iheight + cd.padH)
                    --oheight;
                if ((owidth - 1) * cd.dW >= iwidth + cd.padW)
                    --owidth;
            }

            using (var inputContig = Ops.AsContiguous(input))
            {
                var inputPtr = CudaHelpers.GetBufferStart(inputContig);
                var outputPtr = CudaHelpers.GetBufferStart(output);
                var indicesPtr = CudaHelpers.GetBufferStart(indices);

                var count = (int)output.ElementCount();

                Invoke(context, cudaContext, "MaxPoolForward", new dim3(NNThreads.NumBlocks(count)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                    count, inputPtr, batchSize, nInputPlane, iheight, iwidth, oheight, owidth,
                    cd.kH, cd.kW, cd.dH, cd.dW, cd.padH, cd.padW, outputPtr, indicesPtr);
            }
        }

        public void SpatialMaxPoolingBackward(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var context = CudaHelpers.TSContextForTensor(gradOutput);
            var cudaContext = context.CudaContextForTensor(gradOutput);

            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            var nbatch = input.Sizes[0];
            var nslices = input.Sizes[dimc];
            var iheight = input.Sizes[dimh];
            var iwidth = input.Sizes[dimw];
            var owidth = gradOutput.Sizes[dimw];
            var oheight = gradOutput.Sizes[dimh];


            using (var gradOutputContig = Ops.AsContiguous(gradOutput))
            {
                var gradOutputPtr = CudaHelpers.GetBufferStart(gradOutputContig);
                var indicesPtr = CudaHelpers.GetBufferStart(indices);
                var gradInputPtr = CudaHelpers.GetBufferStart(gradInput);

                var count = (int)input.ElementCount();

                Invoke(context, cudaContext, "MaxPoolBackward", new dim3(NNThreads.NumBlocks(count)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                    count, gradOutputPtr, indicesPtr, nbatch, nslices, iheight, iwidth, oheight, owidth,
                    cd.kH, cd.kW, cd.dH, cd.dW, cd.padH, cd.padW, gradInputPtr);

            }

        }


        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
