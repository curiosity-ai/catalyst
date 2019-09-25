using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.CUDA.DeviceCode;
using Catalyst.Tensors.CUDA.KernelOps;
using Catalyst.Tensors.CUDA.MatrixMul;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA
{
    [OpsClass]
    public class CudaBasicOps
    {
        private readonly CopyOps copyOps;

        private readonly ElementwiseKernels elementwiseKernels = new ElementwiseKernels();
        private readonly FillCopyKernels fillCopyKernels = new FillCopyKernels();

        private readonly CudaReduceKernels cudaReduceKernels = new CudaReduceKernels();
        private readonly CudaReduceAllKernels cudaReduceAllKernels = new CudaReduceAllKernels();

        private readonly VarStdKernels varStdKernels = new VarStdKernels();
        private readonly ReduceDimIndexKernels reduceDimIndexKernels = new ReduceDimIndexKernels();

        private readonly AdvFuncKernels advFuncKernels = new AdvFuncKernels();

        public CudaBasicOps()
        {
            copyOps = new CopyOps(fillCopyKernels);
        }


        /*
        public Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Sizes.Clone());
            Copy(result, src);
            return result;
        }

        public Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
                return src.CopyRef();
            else
                return NewContiguous(src);
        }

        public Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }


        public float SumAll(Tensor src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float ProdAll(Tensor src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MinAll(Tensor src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float MaxAll(Tensor src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        public float MeanAll(Tensor src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float VarAll(Tensor src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float StdAll(Tensor src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public float NormAll(Tensor src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }

        */


        [RegisterOpArgCount("copy")]
        public void CopyGpu(
            [OpArgStorageType(typeof(CudaStorage))] Tensor result,
            [OpArgStorageType(typeof(CudaStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;
            
            copyOps.CopyGpu(result, src, totalElements);
        }

        [RegisterOpArgCount("copy")]
        public void CopyCpuToGpu(
            [OpArgStorageType(typeof(CudaStorage))] Tensor result,
            [OpArgStorageType(typeof(Cpu.CpuStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;

            copyOps.CopyCpuToGpu(result, src, totalElements);
        }

        [RegisterOpArgCount("copy")]
        public void CopyGpuToCpu(
            [OpArgStorageType(typeof(Cpu.CpuStorage))] Tensor result,
            [OpArgStorageType(typeof(CudaStorage))] Tensor src)
        {
            var totalElements = result.ElementCount();
            if (totalElements != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");

            if (src.DimensionCount == 0) return;

            copyOps.CopyGpuToCpu(result, src, totalElements);
        }


        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public void Fill(Tensor result, float value)
        {
            FillOp.Invoke(fillCopyKernels, result, value);
        }


        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return CudaMatrixMulDot.Dot(context, result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return CudaMatrixMulMV.Mul_M_V(context, result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return CudaMatrixMulMM.Mul_M_M(context, result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(m1.Storage is CudaStorage)) throw new ArgumentException("m1 must be a CUDA tensor", "m1");
            if (!(m2.Storage is CudaStorage)) throw new ArgumentException("m2 must be a CUDA tensor", "m2");

            if (src.DimensionCount != 2) throw new ArgumentException("src must be a matrix", "src");
            if (m1.DimensionCount != 2) throw new ArgumentException("m1 must be a matrix", "m1");
            if (m2.DimensionCount != 2) throw new ArgumentException("m2 must be a matrix", "m2");

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }
            
            CudaMatrixMulMM.Gemm(context, alpha, m1, m2, beta, writeTarget);
           

            return writeTarget;
        }



        [RegisterOpStorageType("addmmbatch", typeof(CudaStorage))]
        public Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(m1.Storage is CudaStorage)) throw new ArgumentException("m1 must be a CUDA tensor", "m1");
            if (!(m2.Storage is CudaStorage)) throw new ArgumentException("m2 must be a CUDA tensor", "m2");

            if (src.DimensionCount != 3) throw new ArgumentException("src must be a matrix", "src");
            if (m1.DimensionCount != 3) throw new ArgumentException("m1 must be a matrix", "m1");
            if (m2.DimensionCount != 3) throw new ArgumentException("m2 must be a matrix", "m2");

            if (src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            CudaMatrixMulMM.GemmBatch(context, alpha, m1, m2, beta, writeTarget);


            return writeTarget;
        }

        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public Tensor Abs(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "abs", result, src); }
        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public Tensor Neg(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "neg", result, src); }
        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sign", result, src); }

        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public Tensor Sqrt(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sqrt", result, src); }



        [RegisterOpStorageType("rsqrt", typeof(CudaStorage))]
        public Tensor Rsqrt(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "rsqrt", result, src); }


        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public Tensor Exp(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "exp", result, src); }
        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public Tensor Log(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "log", result, src); }
        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "log1p", result, src); }
        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "floor", result, src); }
        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "ceil", result, src); }
        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public Tensor Round(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "round", result, src); }
        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "trunc", result, src); }
        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "frac", result, src); }

        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sin", result, src); }
        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "cos", result, src); }
        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "tan", result, src); }

        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "asin", result, src); }
        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "acos", result, src); }
        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "atan", result, src); }

        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sinh", result, src); }
        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "cosh", result, src); }
        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public Tensor Tanh(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "tanh", result, src); }

        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src) { return ElementwiseTTOp.Invoke(elementwiseKernels, "sigmoid", result, src); }

        [RegisterOpStorageType("addsigmoidD", typeof(CudaStorage))]
        public Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return ElementwiseTTTTOp.Invoke(elementwiseKernels, "addsigmoidD", result, t, resW, resG); }


        [RegisterOpStorageType("addtanhD", typeof(CudaStorage))]
        public Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return ElementwiseTTTTOp.Invoke(elementwiseKernels, "addtanhD", result, t, resW, resG); }


        [RegisterOpStorageType("sigmoidD", typeof(CudaStorage))]
        public Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "sigmoidD", result, resW, resG); }


        [RegisterOpStorageType("tanhD", typeof(CudaStorage))]
        public Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "tanhD", result, resW, resG); }


        [RegisterOpStorageType("addtanh", typeof(CudaStorage))]
        public Tensor AddTanh(Tensor result, Tensor x, Tensor y) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "addtanh", result, x, y); }


        [RegisterOpStorageType("mulmuladd", typeof(CudaStorage))]
        public Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return ElementwiseTTTTTOp.Invoke(elementwiseKernels, "mulmuladd", result, x, y, z, w); }


        [RegisterOpStorageType("addmul", typeof(CudaStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return ElementwiseTTTTOp.Invoke(elementwiseKernels, "addmul", result, x, y, z); }
        [RegisterOpStorageType("addmulv", typeof(CudaStorage))]
        public Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return ElementwiseTTTSOp.Invoke(elementwiseKernels, "addmulv", result, x, y, z); }


        [RegisterOpStorageType("atan2", typeof(CudaStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return Atan2Op.Invoke(elementwiseKernels, result, srcY, srcX); }
        [RegisterOpStorageType("pow", typeof(CudaStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "pow", result, src, value); }
        [RegisterOpStorageType("tpow", typeof(CudaStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "tpow", result, src, value); }
        [RegisterOpStorageType("lerp", typeof(CudaStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return LerpOp.Invoke(elementwiseKernels, result, srcA, srcB, weight); }
        [RegisterOpStorageType("clamp", typeof(CudaStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max) { return ClampOp.Invoke(elementwiseKernels, result, src, min, max); }

        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public Tensor Add(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "add", result, rhs, lhs); }
        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "sub", result, rhs, lhs); }
        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, float rhs, Tensor lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "rsub", result, lhs, rhs); }
        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public Tensor Mul(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "mul", result, rhs, lhs); }
        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public Tensor Div(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "div", result, rhs, lhs); }
        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public Tensor Div(Tensor result, float rhs, Tensor lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "rdiv", result, lhs, rhs); }
        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public Tensor Mod(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "mod", result, rhs, lhs); }

        [RegisterOpStorageType("gtValue", typeof(CudaStorage))]
        public Tensor GreaterThan(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "gt", result, rhs, lhs); }
        [RegisterOpStorageType("ltValue", typeof(CudaStorage))]
        public Tensor LessThan(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "lt", result, rhs, lhs); }
        [RegisterOpStorageType("geValue", typeof(CudaStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "ge", result, rhs, lhs); }
        [RegisterOpStorageType("leValue", typeof(CudaStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "le", result, rhs, lhs); }
        [RegisterOpStorageType("eqValue", typeof(CudaStorage))]
        public Tensor EqualTo(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "eq", result, rhs, lhs); }
        [RegisterOpStorageType("neValue", typeof(CudaStorage))]
        public Tensor NotEqual(Tensor result, Tensor rhs, float lhs) { return ElementwiseTTSOp.Invoke(elementwiseKernels, "ne", result, rhs, lhs); }


        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public Tensor Add(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cadd", result, rhs, lhs); }
        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public Tensor Sub(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "csub", result, rhs, lhs); }
        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public Tensor Mul(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cmul", result, rhs, lhs); }
        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public Tensor Div(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cdiv", result, rhs, lhs); }
        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public Tensor Mod(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cmod", result, rhs, lhs); }

        [RegisterOpStorageType("gtTensor", typeof(CudaStorage))]
        public Tensor GreaterThan(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cgt", result, rhs, lhs); }
        [RegisterOpStorageType("ltTensor", typeof(CudaStorage))]
        public Tensor LessThan(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "clt", result, rhs, lhs); }
        [RegisterOpStorageType("geTensor", typeof(CudaStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cge", result, rhs, lhs); }
        [RegisterOpStorageType("leTensor", typeof(CudaStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cle", result, rhs, lhs); }
        [RegisterOpStorageType("eqTensor", typeof(CudaStorage))]
        public Tensor EqualTo(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "ceq", result, rhs, lhs); }
        [RegisterOpStorageType("neTensor", typeof(CudaStorage))]
        public Tensor NotEqual(Tensor result, Tensor rhs, Tensor lhs) { return ElementwiseTTTOp.Invoke(elementwiseKernels, "cne", result, rhs, lhs); }


        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "sum", 0.0f, ReduceInitType.GivenValue, result, src, dimension); }
        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "prod", 1.0f, ReduceInitType.GivenValue, result, src, dimension); }
        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "min", 0.0f, ReduceInitType.MaxValue, result, src, dimension); }
        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension) { return ReductionOp.Invoke(cudaReduceKernels, "max", 0.0f, ReduceInitType.MinValue, result, src, dimension); }

        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return reduceDimIndexKernels.ArgMin(result, src, dimension); }

        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension) { return reduceDimIndexKernels.ArgMax(result, src, dimension); }


        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension)
        {
            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, requiredOutputSize);

            Sum(writeTarget, src, dimension);
            Div(writeTarget, writeTarget, src.Sizes[dimension]);
            return writeTarget;
        }

        [RegisterOpStorageType("norm", typeof(CudaStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value)
        {
            if (value == 0)
            {
                return ReductionOp.Invoke(cudaReduceKernels, "e0_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 1)
            {
                return ReductionOp.Invoke(cudaReduceKernels, "e1_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
            }
            else if (value == 2)
            {
                var writeTarget = ReductionOp.Invoke(cudaReduceKernels, "e2_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension);
                Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReductionOp.Invoke(cudaReduceKernels, "en_norm", 0.0f, ReduceInitType.GivenValue, result, src, dimension, value);
                Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }

        [RegisterOpStorageType("std", typeof(CudaStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return varStdKernels.Std(result, src, dimension, normByN); }
        [RegisterOpStorageType("var", typeof(CudaStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return varStdKernels.Var(result, src, dimension, normByN); }


        [RegisterOpStorageType("softmax", typeof(CudaStorage))]
        public Tensor Softmax(Tensor result, Tensor src) { return advFuncKernels.Softmax(result, src); }


        [RegisterOpStorageType("softmaxgrad", typeof(CudaStorage))]
        public Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true) { return advFuncKernels.SoftmaxGrad(grad, adj, val, addGrad); }

        [RegisterOpStorageType("layernorm", typeof(CudaStorage))]
        public Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return advFuncKernels.LayerNorm(result, src, alpha, beta, eps);  }


        [RegisterOpStorageType("layernormgrad", typeof(CudaStorage))]
        public Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps=1e-09f) { return advFuncKernels.LayerNormGrad(outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps); }



        [RegisterOpStorageType("rmsprop", typeof(CudaStorage))]
        public Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            return advFuncKernels.RMSProp(weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);
        }


        [RegisterOpStorageType("sumall", typeof(CudaStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "sumAll", result, src);
        }

        [RegisterOpStorageType("prodall", typeof(CudaStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 1.0f, ReduceInitType.GivenValue, "prodAll", result, src);
        }

        [RegisterOpStorageType("minall", typeof(CudaStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0, ReduceInitType.MaxValue, "minAll", result, src);
        }

        [RegisterOpStorageType("maxall", typeof(CudaStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            return ReduceAllOp.Invoke(cudaReduceAllKernels, 0, ReduceInitType.MinValue, "maxAll", result, src);
        }


        [RegisterOpStorageType("meanall", typeof(CudaStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0) throw new ArgumentException("src must be a non-empty tensor");
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            SumAll(writeTarget, src);
            Div(writeTarget, writeTarget, src.ElementCount());
            return writeTarget;
        }

        [RegisterOpStorageType("normall", typeof(CudaStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            if (value == 0)
            {
                return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e0_norm", result, src);
            }
            else if (value == 1)
            {
                return ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e1_norm", result, src);
            }
            else if (value == 2)
            {
                var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "e2_norm", result, src);
                Pow(writeTarget, writeTarget, 0.5f);
                return writeTarget;
            }
            else
            {
                var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, value);
                Pow(writeTarget, writeTarget, 1.0f / value);
                return writeTarget;
            }
        }


        [RegisterOpStorageType("varall", typeof(CudaStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            if (src.DimensionCount == 0 || src.ElementCount() == 0) throw new ArgumentException("src must be a non-empty tensor");

            var mean = Ops.MeanAll(src);
            var writeTarget = ReduceAllOp.Invoke(cudaReduceAllKernels, 0.0f, ReduceInitType.GivenValue, "en_norm", result, src, mean);
            Div(writeTarget, writeTarget, src.ElementCount() - 1);
            return writeTarget;
        }

        [RegisterOpStorageType("stdall", typeof(CudaStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            var writeTarget = VarAll(result, src);
            Pow(writeTarget, writeTarget, 0.5f);
            return writeTarget;
        }

    }
}
