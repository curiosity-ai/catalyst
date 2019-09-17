//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using Catalyst.Tensors.Cpu;
//using Catalyst.Tensors.CUDA.DeviceCode;

//namespace Catalyst.Tensors.CUDA
//{
//    public class SpatialConvolution
//    {
//        private readonly Im2ColKernels im2colKernels = new Im2ColKernels();

//        public SpatialConvolution()
//        {
//        }

//        public static long[] FInputSize(long[] inputSizes, long[] outputSizes, ConvolutionDesc2d cd)
//        {
//            return new long[] { cd.kW * cd.kH * inputSizes[1], outputSizes[2] * outputSizes[3] };
//        }

//        public void Conv2Forward(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput, ConvolutionDesc2d cd)
//        {
//            var batchSize = input.Sizes[0];
//            var nInputPlane = input.Sizes[1];
//            var inputWidth = input.Sizes[3];
//            var inputHeight = input.Sizes[2];
//            var nOutputPlane = weight.Sizes[0];

//            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
//            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;


//            for (long i = 0; i < batchSize; ++i)
//            {
//                using (var input_i = input.Select(0, i))
//                using (var output_i = output.Select(0, i))
//                {
//                    using (var output2d = output_i.View(nOutputPlane, outputHeight * outputWidth))
//                    {
//                        if (bias != null)
//                        {
//                            using (var biasExp = bias.Expand(nOutputPlane, output2d.Sizes[1]))
//                            {
//                                Ops.Copy(output2d, biasExp);
//                            }
//                        }
//                        else
//                        {
//                            Ops.Fill(output_i, 0);
//                        }

//                        im2colKernels.Im2Col(input_i, finput, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
//                            cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);

//                        Ops.Addmm(output2d, 1, output2d, 1, weight, finput);
//                    }

//                }
//            }
//        }
//        public void Conv2BackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
//        {
//            var nOutputPlane = weight.Sizes[0];
//            var batchSize = input.Sizes[0];

//            var nInputPlane = input.Sizes[1];
//            var inputWidth = input.Sizes[3];
//            var inputHeight = input.Sizes[2];

//            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
//            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;


//            for (long i = 0; i < batchSize; ++i)
//            {
//                using (var gradInput_i = gradInput.Select(0, i))
//                using (var gradOutput_i = gradOutput.Select(0, i))
//                using (var gradOutput_i2d = gradOutput_i.View(nOutputPlane, outputHeight * outputWidth))
//                using (var weightT = weight.Transpose())
//                {
//                    Ops.Addmm(fgradInput, 0, fgradInput, 1, weightT, gradOutput_i2d);

//                    im2colKernels.Col2Im(fgradInput, gradInput_i, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
//                        cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);
//                }
//            }
//        }


//        public void Conv2BackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
//        {
//            var nOutputPlane = gradWeight.Sizes[0];
//            var batchSize = input.Sizes[0];

//            var nInputPlane = input.Sizes[1];
//            var inputWidth = input.Sizes[3];
//            var inputHeight = input.Sizes[2];

//            var outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
//            var outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

//            for (long i = 0; i < batchSize; ++i)
//            {
//                using (var input_i = input.Select(0, i))
//                using (var gradOutput_i = gradOutput.Select(0, i))
//                {
//                    im2colKernels.Im2Col(input_i, finput, (int)nInputPlane, (int)inputHeight, (int)inputWidth,
//                        cd.kH, cd.kW, cd.padH, cd.padW, cd.dH, cd.dW, 1, 1);

//                    using (var gradOutput2d = gradOutput_i.View(gradOutput_i.Sizes[0], gradOutput_i.Sizes[1] * gradOutput_i.Sizes[2]))
//                    using (var finputT = finput.Transpose())
//                    {
//                        Ops.Addmm(gradWeight, 1, gradWeight, 1, gradOutput2d, finputT);
//                        Ops.Sum(gradBias, gradOutput2d, 1);
//                    }

//                }
//            }
//        }
//    }
//}
