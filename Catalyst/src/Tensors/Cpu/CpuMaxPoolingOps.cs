using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    public static class CpuMaxPoolingOps
    {
        public static long[] OutputSize(long[] inputSizes, bool ceilMode, ConvolutionDesc2d cd)
        {
            int dimw = 3;
            int dimh = 2;

            var iwidth = inputSizes[dimw];
            var iheight = inputSizes[dimh];

            long oheight, owidth;
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

            return new long[] { inputSizes[0], inputSizes[1], oheight, owidth };
        }


        public static void SpatialMaxPoolingForward(Tensor input, Tensor output, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            if (input.DimensionCount != 4) throw new ArgumentException("input must be a 4D tensor");

            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            if (input.Sizes[dimw] < cd.kW - cd.padW || input.Sizes[dimh] < cd.kH - cd.padH)
                throw new InvalidOperationException("input image is smaller than kernel size");

            if (cd.padW > cd.kW / 2 || cd.padH > cd.kH / 2)
                throw new InvalidOperationException("pad should be smaller than half of the kernel size");

            var nbatch = input.Sizes[0];
            var nslices = input.Sizes[dimc];
            var iheight = input.Sizes[dimh];
            var iwidth = input.Sizes[dimw];

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
                for (int i = 0; i < nbatch; ++i)
                {
                    using (var input_i = inputContig.Select(0, i))
                    using (var output_i = output.Select(0, i))
                    using (var indices_i = indices.Select(0, i))
                    {
                        IntPtr input_iPtr, output_iPtr, indices_iPtr;
                        using (NativeWrapper.BuildTensorRefPtr(input_i, out input_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(output_i, out output_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(indices_i, out indices_iPtr))
                        {
                            CpuOpsNative.TS_SpatialMaxPooling_updateOutput_frame(input_iPtr, output_iPtr, indices_iPtr,
                                nslices, iwidth, iheight,
                                owidth, oheight,
                                cd.kW, cd.kH, cd.dW, cd.dH, cd.padW, cd.padH);
                        }
                    }
                }
            }

        }


        public static void SpatialMaxPoolingBackward(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            var nbatch = input.Sizes[0];
            var nslices = input.Sizes[dimc];
            var iheight = input.Sizes[dimh];
            var iwidth = input.Sizes[dimw];
            var owidth = gradOutput.Sizes[dimw];
            var oheight = gradOutput.Sizes[dimh];

            Ops.Fill(gradInput, 0);


            using (var gradOutputContig = Ops.AsContiguous(gradOutput))
            {
                for (int i = 0; i < nbatch; ++i)
                {
                    using (var gradInput_i = gradInput.Select(0, i))
                    using (var gradOutput_i = gradOutputContig.Select(0, i))
                    using (var indices_i = indices.Select(0, i))
                    {
                        IntPtr gradInput_iPtr, gradOutput_iPtr, indices_iPtr;
                        using (NativeWrapper.BuildTensorRefPtr(gradInput_i, out gradInput_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(gradOutput_i, out gradOutput_iPtr))
                        using (NativeWrapper.BuildTensorRefPtr(indices_i, out indices_iPtr))
                        {
                            CpuOpsNative.TS_SpatialMaxPooling_updateGradInput_frame(gradInput_iPtr, gradOutput_iPtr, indices_iPtr,
                                nslices, iwidth, iheight,
                                owidth, oheight,
                                cd.dW, cd.dH);
                        }
                    }
                }
            }
        }
    }
}
