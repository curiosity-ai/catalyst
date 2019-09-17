using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors
{
    public static class Ops
    {
        public static Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Sizes.Clone());
            Copy(result, src);
            return result;
        }

        public static Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
                return src.CopyRef();
            else
                return NewContiguous(src);
        }

        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }

        public static void FillOneHot(Tensor result, int labelCount, int[] labels)
        {
            if (result.Storage is Cpu.CpuStorage)
            {
                DoFillOneHot(result, labelCount, labels);
            }
            else
            {
                //If the result is not on the CPU, it is much faster to build the tensor on the CPU and then copy
                //An alternative to this would be building a specific GPU kernel for this operation
                var cpuAlloc = new Cpu.CpuAllocator();
                using (var cpuResult = new Tensor(cpuAlloc, result.ElementType, result.Sizes))
                {
                    DoFillOneHot(cpuResult, labelCount, labels);
                    Ops.Copy(result, cpuResult);
                }
            }
        }

        private static void DoFillOneHot(Tensor result, int labelCount, int[] labels)
        {
            if (result.DimensionCount != 2) throw new InvalidOperationException("result must be a 2D tensor");
            if (result.Sizes[0] != labels.Length) throw new InvalidOperationException("first dimension of result must equal the number of samples");
            if (result.Sizes[1] > labelCount) throw new InvalidOperationException("second dimension of result must be at least as large as labelCount");

            Ops.Fill(result, 0);
            for (int i = 0; i < labels.Length; ++i)
            {
                if (labels[i] < 0 || labels[i] >= labelCount)
                    throw new InvalidOperationException("label at index " + i + " is out of range 0 <= x < labelCount");

                result.SetElementAsFloat(1.0f, i, labels[i]);
            }
        }




        public static void Copy(Tensor result, Tensor src) { OpRegistry.Invoke("copy", result, src); }
        public static void Fill(Tensor result, float value) { OpRegistry.Invoke("fill", result, value); }

        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("dot", result, lhs, rhs); }
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmm", result, beta, src, alpha, m1, m2); }

        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmmbatch", result, beta, src, alpha, m1, m2); }

        public static Tensor Abs(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("abs", result, src); }
        public static Tensor Neg(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("neg", result, src); }
        public static Tensor Sign(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sign", result, src); }

        public static Tensor Sqrt(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sqrt", result, src); }

        public static Tensor Rsqrt(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("rsqrt", result, src); }

        public static Tensor Exp(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("exp", result, src); }
        public static Tensor Log(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log", result, src); }
        public static Tensor Log1p(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log1p", result, src); }
        public static Tensor Floor(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("floor", result, src); }
        public static Tensor Ceil(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("ceil", result, src); }
        public static Tensor Round(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("round", result, src); }
        public static Tensor Trunc(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("trunc", result, src); }
        public static Tensor Frac(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("frac", result, src); }

        public static Tensor Sin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sin", result, src); }
        public static Tensor Cos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cos", result, src); }
        public static Tensor Tan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tan", result, src); }

        public static Tensor Asin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("asin", result, src); }
        public static Tensor Acos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("acos", result, src); }
        public static Tensor Atan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("atan", result, src); }

        public static Tensor Sinh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sinh", result, src); }
        public static Tensor Cosh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cosh", result, src); }
        public static Tensor Tanh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tanh", result, src); }

        public static Tensor Sigmoid(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sigmoid", result, src); }

        public static Tensor AddSigmoidD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("addsigmoidD", result, t, resW, resG); }

        public static Tensor AddTanhD(Tensor result, Tensor t, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("addtanhD", result, t, resW, resG); }


        public static Tensor SigmoidD(Tensor result, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("sigmoidD", result, resW, resG); }

        public static Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return (Tensor)OpRegistry.Invoke("tanhD", result, resW, resG); }



        public static Tensor AddTanh(Tensor result, Tensor x, Tensor y) { return (Tensor)OpRegistry.Invoke("addtanh", result, x, y); }

        public static Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return (Tensor)OpRegistry.Invoke("mulmuladd", result, x, y, z, w); }

        public static Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("addmul", result, x, y, z); }
        public static Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return (Tensor)OpRegistry.Invoke("addmulv", result, x, y, z); }

        public static Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return (Tensor)OpRegistry.Invoke("atan2", result, srcY, srcX); }
        public static Tensor Pow(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("pow", result, src, value); }
        public static Tensor Tpow(Tensor result, float value, Tensor src) { return (Tensor)OpRegistry.Invoke("tpow", result, value, src); }
        public static Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return (Tensor)OpRegistry.Invoke("lerp", result, srcA, srcB); }
        public static Tensor Clamp(Tensor result, Tensor src, float min, float max) { return (Tensor)OpRegistry.Invoke("clamp", result, src, min, max); }


        public static Tensor Add(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("addv", result, lhs, rhs); }
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("subv", result, lhs, rhs); }
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rsubv", result, lhs, rhs); }
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("mulv", result, lhs, rhs); }
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("divv", result, lhs, rhs); }
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rdivv", result, lhs, rhs); }
        public static Tensor Mod(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("modv", result, lhs, rhs); }

        public static Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("gtValue", result, lhs, rhs); }
        public static Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("ltValue", result, lhs, rhs); }
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("geValue", result, lhs, rhs); }
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("leValue", result, lhs, rhs); }
        public static Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("eqValue", result, lhs, rhs); }
        public static Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("neValue", result, lhs, rhs); }

        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("addt", result, lhs, rhs); }
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("subt", result, lhs, rhs); }
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("mult", result, lhs, rhs); }
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("divt", result, lhs, rhs); }
        public static Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("modt", result, lhs, rhs); }

        public static Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("gtTensor", result, lhs, rhs); }
        public static Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("ltTensor", result, lhs, rhs); }
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("geTensor", result, lhs, rhs); }
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("leTensor", result, lhs, rhs); }
        public static Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("eqTensor", result, lhs, rhs); }
        public static Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("neTensor", result, lhs, rhs); }


        public static Tensor Sum(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("sum", result, src, dimension); }
        public static Tensor Prod(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("prod", result, src, dimension); }
        public static Tensor Min(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("min", result, src, dimension); }
        public static Tensor Max(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("max", result, src, dimension); }
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmin", result, src, dimension); }
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmax", result, src, dimension); }

        public static Tensor Mean(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("mean", result, src, dimension); }
        public static Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return (Tensor)OpRegistry.Invoke("norm", result, src, dimension, value); }
        public static Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("std", result, src, dimension, normByN); }
        public static Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("var", result, src, dimension, normByN); }


        public static Tensor Softmax(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("softmax", result, src); }

        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad = true) { return (Tensor)OpRegistry.Invoke("softmaxgrad", grad, adj, val, addGrad); }



        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("layernorm", result, src, alpha, beta, eps); }


        public static Tensor LayerNormGrad(Tensor outGrad, Tensor alphaGrad, Tensor betaGrad, Tensor inGrad, Tensor y, Tensor x, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("layernormgrad", outGrad, alphaGrad, betaGrad, inGrad, y, x, alpha, beta, eps); }


        public static Tensor RMSProp(Tensor weight, Tensor gradient, Tensor cache, int batchSize, float step_size, float clipval, float regc, float decay_rate, float eps)
        {
            return (Tensor)OpRegistry.Invoke("rmsprop", weight, gradient, cache, batchSize, step_size, clipval, regc, decay_rate, eps);
        }

        public static Tensor SumAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sumall", result, src); }
        public static Tensor ProdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("prodall", result, src); }
        public static Tensor MinAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("minall", result, src); }
        public static Tensor MaxAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("maxall", result, src); }

        public static Tensor MeanAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("meanall", result, src); }
        public static Tensor NormAll(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("normall", result, src, value); }
        public static Tensor StdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("stdall", result, src); }
        public static Tensor VarAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("varall", result, src); }


        public static float SumAll(Tensor src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float ProdAll(Tensor src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float MinAll(Tensor src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float MaxAll(Tensor src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        public static float MeanAll(Tensor src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float VarAll(Tensor src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float StdAll(Tensor src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float NormAll(Tensor src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }


        public static Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("index_select", result, src, dim, indices); }
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("gather", result, src, dim, indices); }
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter", result, src, dim, indices); }
        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_fill", result, value, dim, indices); }


        private static int? GetSeed(SeedSource src)
        {
            return src == null ? (int?)null : src.NextSeed();
        }

        public static void RandomUniform(Tensor result, SeedSource seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", result, GetSeed(seedSource), min, max); }
        public static void RandomNormal(Tensor result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", result, GetSeed(seedSource), mean, stdv); }
        public static void RandomExponential(Tensor result, SeedSource seedSource, float lambda) { OpRegistry.Invoke("random_exponential", result, GetSeed(seedSource), lambda); }
        public static void RandomCauchy(Tensor result, SeedSource seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", result, GetSeed(seedSource), median, sigma); }
        public static void RandomLogNormal(Tensor result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", result, GetSeed(seedSource), mean, stdv); }
        public static void RandomGeometric(Tensor result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_geometric", result, GetSeed(seedSource), p); }
        public static void RandomBernoulli(Tensor result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_bernoulli", result, GetSeed(seedSource), p); }
    }
}
