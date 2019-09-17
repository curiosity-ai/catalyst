//using ManagedCuda;
//using ManagedCuda.CudaDNN;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using Catalyst.Tensors.Core;

//namespace Catalyst.Tensors.CUDA
//{
//    public class TensorShape
//    {
//        public DType ElementType { get; private set; }
//        public long[] Sizes { get; private set; }
//        public long[] Strides { get; private set; }
//        public int DimensionCount { get { return Sizes.Length; } }

//        public TensorShape(Tensor tensor)
//        {
//            this.ElementType = tensor.ElementType;
//            this.Sizes = tensor.Sizes;
//            this.Strides = tensor.Strides;
//        }

//        public TensorShape(DType elementType, long[] sizes, long[] strides)
//        {
//            this.ElementType = elementType;
//            this.Sizes = sizes;
//            this.Strides = strides;
//        }

//        public TensorShape(DType elementType, long[] sizes)
//            : this(elementType, sizes, TensorDimensionHelpers.GetContiguousStride(sizes))
//        {
//        }
//    }

//    public enum DNNActivation
//    {
//        Sigmoid = cudnnActivationMode.Sigmoid,
//        Relu = cudnnActivationMode.Relu,
//        Tanh = cudnnActivationMode.Tanh,
//        ClippedRelu = cudnnActivationMode.ClippedRelu,
//    }

//    public enum DNNSoftmaxAlgorithm
//    {
//        Fast = cudnnSoftmaxAlgorithm.Fast,
//        Accurate = cudnnSoftmaxAlgorithm.Accurate,
//        Log = cudnnSoftmaxAlgorithm.Log,
//    }

//    public enum DNNSoftmaxMode
//    {
//        Instance = cudnnSoftmaxMode.Instance,
//        Channel = cudnnSoftmaxMode.Channel,
//    }

//    public enum DNNPoolingMode
//    {
//        Max = cudnnPoolingMode.Max,
//        AverageCountIncludePadding = cudnnPoolingMode.AverageCountIncludePadding,
//        AverageCountExcludePadding = cudnnPoolingMode.AverageCountExcludePadding,
//    }

//    public enum DNNConvolutionFwdAlgo
//    {
//        ImplicitGEMM = cudnnConvolutionFwdAlgo.ImplicitGEMM,
//        ImplicitPrecompGEMM = cudnnConvolutionFwdAlgo.ImplicitPrecompGEMM,
//        GEMM = cudnnConvolutionFwdAlgo.GEMM,
//        Direct = cudnnConvolutionFwdAlgo.Direct,
//        FFT = cudnnConvolutionFwdAlgo.FFT,
//        FFTWithTiling = cudnnConvolutionFwdAlgo.FFTWithTiling,
//        Winograd = cudnnConvolutionFwdAlgo.Winograd,
//    }

//    public enum DNNConvolutionBwdFilterAlgo
//    {
//        Algo0 = cudnnConvolutionBwdFilterAlgo.Algo0,
//        Algo1 = cudnnConvolutionBwdFilterAlgo.Algo1,
//        Algo3 = cudnnConvolutionBwdFilterAlgo.Algo3,
//        AlgoFFT = cudnnConvolutionBwdFilterAlgo.AlgoFFT,
//    }

//    public enum DNNConvolutionBwdDataAlgo
//    {
//        Algo0 = cudnnConvolutionBwdDataAlgo.Algo0,
//        Algo1 = cudnnConvolutionBwdDataAlgo.Algo1,
//        AlgoFFT = cudnnConvolutionBwdDataAlgo.AlgoFFT,
//        Winograd = cudnnConvolutionBwdDataAlgo.Winograd,
//    }


//    public struct DNNPoolingDesc
//    {
//        public DNNPoolingMode Mode;
//        public int[] WindowDims;
//        public int[] Padding;
//        public int[] Strides;

//        public DNNPoolingDesc(DNNPoolingMode mode, int[] windowDims, int[] padding, int[] strides)
//        {
//            this.Mode = mode;
//            this.WindowDims = windowDims;
//            this.Padding = padding;
//            this.Strides = strides;
//        }

//        public DNNPoolingDesc(DNNPoolingMode mode, int dimA, int dimB, int padA, int padB, int strideA, int strideB)
//        {
//            this.Mode = mode;
//            this.WindowDims = new int[] { dimA, dimB };
//            this.Padding = new int[] { padA, padB };
//            this.Strides = new int[] { strideA, strideB };
//        }
//    }


//    public static class DNN
//    {
//        public static void ActivationForward(Tensor x, Tensor y, DNNActivation activationType, double clippedReluCeiling)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {

//                var activationDesc = new ActivationDescriptor();
//                activationDesc.SetActivationDescriptor((cudnnActivationMode)activationType,
//                    cudnnNanPropagation.PropagateNan,
//                    clippedReluCeiling);

//                using (var xPtr = GetDeviceVar(x))
//                using (var yPtr = GetDeviceVar(y))
//                using (var xDesc = GetDescriptor(x))
//                using (var yDesc = GetDescriptor(y))
//                {
//                    dnn.Value.ActivationForward(activationDesc, 1,
//                        xDesc, xPtr,
//                        0,
//                        yDesc, yPtr);
//                }
//            }
//        }

//        public static void ActivationBackward(Tensor x, Tensor y, Tensor dx, Tensor dy, DNNActivation activationType, double clippedReluCeiling)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {

//                var activationDesc = new ActivationDescriptor();
//                activationDesc.SetActivationDescriptor((cudnnActivationMode)activationType,
//                    cudnnNanPropagation.PropagateNan,
//                    clippedReluCeiling);

//                using (var xPtr = GetDeviceVar(x))
//                using (var yPtr = GetDeviceVar(y))
//                using (var dxPtr = GetDeviceVar(dx))
//                using (var dyPtr = GetDeviceVar(dy))
//                using (var xDesc = GetDescriptor(x))
//                using (var yDesc = GetDescriptor(y))
//                using (var dxDesc = GetDescriptor(dx))
//                using (var dyDesc = GetDescriptor(dy))
//                {
//                    dnn.Value.ActivationBackward(activationDesc, 1,
//                        xDesc, xPtr,
//                        dxDesc, dxPtr,
//                        yDesc, yPtr,
//                        0,
//                        dyDesc, dyPtr);
//                }
//            }
//        }

//        public static void SoftmaxForward(DNNSoftmaxAlgorithm algorithm, DNNSoftmaxMode mode, Tensor x, Tensor y)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {

//                using (var xPtr = GetDeviceVar(x))
//                using (var yPtr = GetDeviceVar(y))
//                using (var xDesc = GetDescriptor(x))
//                using (var yDesc = GetDescriptor(y))
//                {
//                    dnn.Value.SoftmaxForward((cudnnSoftmaxAlgorithm)algorithm, (cudnnSoftmaxMode)mode, 1,
//                    xDesc, xPtr,
//                    0,
//                    yDesc, yPtr);
//                }
//            }
//        }

//        public static void SoftmaxBackward(DNNSoftmaxAlgorithm algorithm, DNNSoftmaxMode mode, Tensor y, Tensor dx, Tensor dy)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(y).DNNForTensor(y))
//            {

//                using (var yPtr = GetDeviceVar(y))
//                using (var dxPtr = GetDeviceVar(dx))
//                using (var dyPtr = GetDeviceVar(dy))
//                using (var yDesc = GetDescriptor(y))
//                using (var dxDesc = GetDescriptor(dx))
//                using (var dyDesc = GetDescriptor(dy))
//                {
//                    dnn.Value.SoftmaxBackward((cudnnSoftmaxAlgorithm)algorithm, (cudnnSoftmaxMode)mode, 1,
//                    yDesc, yPtr,
//                    dyDesc, dyPtr,
//                    0,
//                    dxDesc, dxPtr);
//                }
//            }
//        }

//        public static void PoolingForward(DNNPoolingDesc desc, Tensor x, Tensor y)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {

//                var poolingDesc = new PoolingDescriptor();
//                poolingDesc.SetPoolingNdDescriptor((cudnnPoolingMode)desc.Mode, cudnnNanPropagation.PropagateNan, desc.WindowDims.Length,
//                    desc.WindowDims, desc.Padding, desc.Strides);

//                using (var xPtr = GetDeviceVar(x))
//                using (var yPtr = GetDeviceVar(y))
//                using (var xDesc = GetDescriptor(x))
//                using (var yDesc = GetDescriptor(y))
//                {
//                    dnn.Value.PoolingForward(poolingDesc, 1,
//                        xDesc, xPtr,
//                        0,
//                        yDesc, yPtr);
//                }
//            }
//        }

//        public static void PoolingBackward(DNNPoolingDesc desc, Tensor x, Tensor y, Tensor dx, Tensor dy)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {
//                var poolingDesc = new PoolingDescriptor();
//                poolingDesc.SetPoolingNdDescriptor((cudnnPoolingMode)desc.Mode, cudnnNanPropagation.PropagateNan, desc.WindowDims.Length,
//                    desc.WindowDims, desc.Padding, desc.Strides);

//                using (var xPtr = GetDeviceVar(x))
//                using (var yPtr = GetDeviceVar(y))
//                using (var dxPtr = GetDeviceVar(dx))
//                using (var dyPtr = GetDeviceVar(dy))
//                using (var xDesc = GetDescriptor(x))
//                using (var yDesc = GetDescriptor(y))
//                using (var dxDesc = GetDescriptor(dx))
//                using (var dyDesc = GetDescriptor(dy))
//                {
//                    // Note: ManagedCUDA argument names may be slightly misleading (src refers to 'y' here, and dest to 'x')
//                    dnn.Value.PoolingBackward(poolingDesc, 1,
//                        yDesc, yPtr,
//                        dyDesc, dyPtr,
//                        xDesc, xPtr,
//                        0,
//                        dxDesc, dxPtr);
//                }
//            }
//        }



//        public static void AddTensor(Tensor src, Tensor result)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(src).DNNForTensor(src))
//            {

//                using (var srcPtr = GetDeviceVar(src))
//                using (var resultPtr = GetDeviceVar(result))
//                using (var srcDesc = GetDescriptor(src))
//                using (var resultDesc = GetDescriptor(result))
//                {
//                    dnn.Value.AddTensor(1,
//                        srcDesc, srcPtr,
//                        1,
//                        resultDesc, resultPtr);
//                }
//            }
//        }

//        private static ConvolutionDescriptor GetConvDescriptor(Cpu.ConvolutionDesc2d cd, DType elementType)
//        {
//            var convDesc = new ConvolutionDescriptor();
//            convDesc.SetConvolution2dDescriptor(cd.padH, cd.padW, cd.dH, cd.dW, 1, 1, cudnnConvolutionMode.CrossCorrelation, GetDataType(elementType));
//            return convDesc;
//        }

//        public static void ConvForward(DNNConvolutionFwdAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, Tensor x, Tensor w, Tensor y)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {
//                var convDesc = GetConvDescriptor(cd, x.ElementType);

//                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
//                using (var xPtr = GetDeviceVar(x))
//                using (var wPtr = GetDeviceVar(w))
//                using (var yPtr = GetDeviceVar(y))
//                using (var xDesc = GetDescriptor(x))
//                using (var wDesc = GetFilterDescriptor(w))
//                using (var yDesc = GetDescriptor(y))
//                {
//                    dnn.Value.ConvolutionForward(1,
//                        xDesc, xPtr,
//                        wDesc, wPtr,
//                        convDesc,
//                        (cudnnConvolutionFwdAlgo)algo,
//                        workspacePtr,
//                        0,
//                        yDesc, yPtr);
//                }
//            }
//        }

//        public static void ConvolutionBackwardData(DNNConvolutionBwdDataAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, Tensor w, Tensor dy, Tensor dx)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(w).DNNForTensor(w))
//            {

//                var convDesc = GetConvDescriptor(cd, w.ElementType);

//                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
//                using (var wPtr = GetDeviceVar(w))
//                using (var dxPtr = GetDeviceVar(dx))
//                using (var dyPtr = GetDeviceVar(dy))
//                using (var wDesc = GetFilterDescriptor(w))
//                using (var dxDesc = GetDescriptor(dx))
//                using (var dyDesc = GetDescriptor(dy))
//                {
//                    dnn.Value.ConvolutionBackwardData(1,
//                        wDesc, wPtr,
//                        dyDesc, dyPtr,
//                        convDesc,
//                        0,
//                        (cudnnConvolutionBwdDataAlgo)algo,
//                        workspacePtr,
//                        dxDesc, dxPtr);
//                }
//            }
//        }

//        public static void ConvolutionBackwardFilter(DNNConvolutionBwdFilterAlgo algo, Cpu.ConvolutionDesc2d cd, CudaStorage workspace, Tensor x, Tensor dy, Tensor dw)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(x).DNNForTensor(x))
//            {
//                var convDesc = GetConvDescriptor(cd, x.ElementType);

//                using (var workspacePtr = new CudaDeviceVariable<byte>(workspace.DevicePtrAtElement(0), false, workspace.ByteLength))
//                using (var xPtr = GetDeviceVar(x))
//                using (var dyPtr = GetDeviceVar(dy))
//                using (var dwPtr = GetDeviceVar(dw))
//                using (var xDesc = GetDescriptor(x))
//                using (var dyDesc = GetDescriptor(dy))
//                using (var dwDesc = GetFilterDescriptor(dw))
//                {
//                    dnn.Value.ConvolutionBackwardFilter(1,
//                        xDesc, xPtr,
//                        dyDesc, dyPtr,
//                        convDesc,
//                        (cudnnConvolutionBwdFilterAlgo)algo,
//                        workspacePtr,
//                        0,
//                        dwDesc, dwPtr);
//                }
//            }
//        }


//        public static void ConvolutionBackwardBias(Cpu.ConvolutionDesc2d cd, Tensor dy, Tensor db)
//        {
//            using (var dnn = CudaHelpers.TSContextForTensor(dy).DNNForTensor(dy))
//            {

//                using (var dyPtr = GetDeviceVar(dy))
//                using (var dbPtr = GetDeviceVar(db))
//                using (var dyDesc = GetDescriptor(dy))
//                using (var dbDesc = GetDescriptor(db))
//                {
//                    dnn.Value.ConvolutionBackwardBias(1,
//                        dyDesc, dyPtr,
//                        0,
//                        dbDesc, dbPtr);
//                }
//            }
//        }


//        public static long GetConvolutionForwardWorkspaceSize(IAllocator allocator, DNNConvolutionFwdAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape x, TensorShape w, TensorShape y)
//        {
//            if (!(allocator is CudaAllocator))
//                throw new InvalidOperationException("allocator must be a CUDA allocator");

//            var cudaAllocator = (CudaAllocator)allocator;

//            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
//            {

//                var convDesc = GetConvDescriptor(cd, x.ElementType);

//                using (var xDesc = GetDescriptor(x))
//                using (var wDesc = GetFilterDescriptor(w))
//                using (var yDesc = GetDescriptor(y))
//                {
//                    return dnn.Value.GetConvolutionForwardWorkspaceSize(
//                        xDesc,
//                        wDesc,
//                        convDesc,
//                        yDesc,
//                        (cudnnConvolutionFwdAlgo)algo);
//                }
//            }
//        }

//        public static long GetConvolutionBackwardFilterWorkspaceSize(IAllocator allocator, DNNConvolutionBwdFilterAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape x, TensorShape dy, TensorShape dw)
//        {
//            if (!(allocator is CudaAllocator))
//                throw new InvalidOperationException("allocator must be a CUDA allocator");

//            var cudaAllocator = (CudaAllocator)allocator;

//            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
//            {

//                var convDesc = GetConvDescriptor(cd, x.ElementType);

//                using (var xDesc = GetDescriptor(x))
//                using (var dyDesc = GetDescriptor(dy))
//                using (var dwDesc = GetFilterDescriptor(dw))
//                {
//                    return dnn.Value.GetConvolutionBackwardFilterWorkspaceSize(
//                        xDesc,
//                        dyDesc,
//                        convDesc,
//                        dwDesc,
//                        (cudnnConvolutionBwdFilterAlgo)algo);
//                }
//            }
//        }

//        public static long GetConvolutionBackwardDataWorkspaceSize(IAllocator allocator, DNNConvolutionBwdDataAlgo algo, Cpu.ConvolutionDesc2d cd, TensorShape w, TensorShape dy, TensorShape dx)
//        {
//            if (!(allocator is CudaAllocator))
//                throw new InvalidOperationException("allocator must be a CUDA allocator");

//            var cudaAllocator = (CudaAllocator)allocator;

//            using (var dnn = cudaAllocator.Context.DNNForDevice(cudaAllocator.DeviceId))
//            {

//                var convDesc = GetConvDescriptor(cd, w.ElementType);

//                using (var wDesc = GetFilterDescriptor(w))
//                using (var dyDesc = GetDescriptor(dy))
//                using (var dxDesc = GetDescriptor(dx))
//                {
//                    return dnn.Value.GetConvolutionBackwardDataWorkspaceSize(
//                        wDesc,
//                        dyDesc,
//                        convDesc,
//                        dxDesc,
//                        (cudnnConvolutionBwdDataAlgo)algo);
//                }
//            }
//        }
        

//        private static CudaDeviceVariable<float> GetDeviceVar(Tensor tensor)
//        {
//            var ptr = CudaHelpers.GetBufferStart(tensor);
//            return new CudaDeviceVariable<float>(ptr, false, 0);// set size to 0 because we never end up using it
//        }

//        private static TensorDescriptor GetDescriptor(Tensor tensor)
//        {
//            var result = new TensorDescriptor();
//            result.SetTensorNdDescriptor(
//                GetDataType(tensor.ElementType),
//                tensor.DimensionCount,
//                tensor.Sizes.Select(x => (int)x).ToArray(),
//                tensor.Strides.Select(x => (int)x).ToArray());
//            return result;
//        }

//        private static FilterDescriptor GetFilterDescriptor(Tensor tensor)
//        {
//            var result = new FilterDescriptor();
//            result.SetFilterNdDescriptor(
//                GetDataType(tensor.ElementType),
//                cudnnTensorFormat.NCHW,
//                tensor.DimensionCount,
//                tensor.Sizes.Select(x => (int)x).ToArray());
//            return result;
//        }

//        private static TensorDescriptor GetDescriptor(TensorShape shape)
//        {
//            var result = new TensorDescriptor();
//            result.SetTensorNdDescriptor(
//                GetDataType(shape.ElementType),
//                shape.DimensionCount,
//                shape.Sizes.Select(x => (int)x).ToArray(),
//                shape.Strides.Select(x => (int)x).ToArray());
//            return result;
//        }

//        private static FilterDescriptor GetFilterDescriptor(TensorShape shape)
//        {
//            var result = new FilterDescriptor();
//            result.SetFilterNdDescriptor(
//                GetDataType(shape.ElementType),
//                cudnnTensorFormat.NCHW,
//                shape.DimensionCount,
//                shape.Sizes.Select(x => (int)x).ToArray());
//            return result;
//        }


//        private static cudnnDataType GetDataType(DType dataType)
//        {
//            switch(dataType)
//            {
//                case DType.Float32: return cudnnDataType.Float;
//                case DType.Float64: return cudnnDataType.Double;
//                case DType.Float16: return cudnnDataType.Half;
//                default:
//                    throw new NotSupportedException("DNN: type not supported: " + dataType);
//            }
//        }
//    }
//}
