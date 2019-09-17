using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors.Cpu
{
    [OpsClass]
    public class CpuBasicOps
    {
        public CpuBasicOps()
        {
        }
        

        [RegisterOpStorageType("dot", typeof(CpuStorage))]
        public Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Dot(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Mul_M_V(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return MatrixMultiplication.Mul_M_M(result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CpuStorage))]
        public Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CpuStorage)) throw new ArgumentException("result must be a CPU tensor", "result");
            if (!(m1.Storage is CpuStorage)) throw new ArgumentException("m1 must be a CPU tensor", "m1");
            if (!(m2.Storage is CpuStorage)) throw new ArgumentException("m2 must be a CPU tensor", "m2");

            if (src.DimensionCount != 2) throw new ArgumentException("src must be a matrix", "src");
            if (m1.DimensionCount != 2) throw new ArgumentException("m1 must be a matrix", "m1");
            if (m2.DimensionCount != 2) throw new ArgumentException("m2 must be a matrix", "m2");

            if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
                throw new InvalidOperationException("Size mismatch");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            
            MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);
            

            return writeTarget;
        }


        


        private MethodInfo abs_func = NativeWrapper.GetMethod("TS_Abs");
        [RegisterOpStorageType("abs", typeof(CpuStorage))]
        public Tensor Abs(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(abs_func, result, src); }

        private MethodInfo neg_func = NativeWrapper.GetMethod("TS_Neg");
        [RegisterOpStorageType("neg", typeof(CpuStorage))]
        public Tensor Neg(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(neg_func, result, src); }

        private MethodInfo sign_func = NativeWrapper.GetMethod("TS_Sign");
        [RegisterOpStorageType("sign", typeof(CpuStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sign_func, result, src); }


        private MethodInfo sqrt_func = NativeWrapper.GetMethod("TS_Sqrt");
        [RegisterOpStorageType("sqrt", typeof(CpuStorage))]
        public Tensor Sqrt(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sqrt_func, result, src); }


        private MethodInfo rsqrt_func = NativeWrapper.GetMethod("TS_Rsqrt");
        [RegisterOpStorageType("rsqrt", typeof(CpuStorage))]
        public Tensor Rsqrt(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(rsqrt_func, result, src); }

        private MethodInfo exp_func = NativeWrapper.GetMethod("TS_Exp");
        [RegisterOpStorageType("exp", typeof(CpuStorage))]
        public Tensor Exp(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(exp_func, result, src); }

        private MethodInfo log_func = NativeWrapper.GetMethod("TS_Log");
        [RegisterOpStorageType("log", typeof(CpuStorage))]
        public Tensor Log(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log_func, result, src); }

        private MethodInfo log1p_func = NativeWrapper.GetMethod("TS_Log1p");
        [RegisterOpStorageType("log1p", typeof(CpuStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log1p_func, result, src); }

        private MethodInfo floor_func = NativeWrapper.GetMethod("TS_Floor");
        [RegisterOpStorageType("floor", typeof(CpuStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(floor_func, result, src); }

        private MethodInfo ceil_func = NativeWrapper.GetMethod("TS_Ceil");
        [RegisterOpStorageType("ceil", typeof(CpuStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(ceil_func, result, src); }

        private MethodInfo round_func = NativeWrapper.GetMethod("TS_Round");
        [RegisterOpStorageType("round", typeof(CpuStorage))]
        public Tensor Round(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(round_func, result, src); }

        private MethodInfo trunc_func = NativeWrapper.GetMethod("TS_Trunc");
        [RegisterOpStorageType("trunc", typeof(CpuStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(trunc_func, result, src); }

        private MethodInfo frac_func = NativeWrapper.GetMethod("TS_Frac");
        [RegisterOpStorageType("frac", typeof(CpuStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(frac_func, result, src); }


        private MethodInfo sin_func = NativeWrapper.GetMethod("TS_Sin");
        [RegisterOpStorageType("sin", typeof(CpuStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sin_func, result, src); }

        private MethodInfo cos_func = NativeWrapper.GetMethod("TS_Cos");
        [RegisterOpStorageType("cos", typeof(CpuStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cos_func, result, src); }

        private MethodInfo tan_func = NativeWrapper.GetMethod("TS_Tan");
        [RegisterOpStorageType("tan", typeof(CpuStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tan_func, result, src); }


        private MethodInfo asin_func = NativeWrapper.GetMethod("TS_Asin");
        [RegisterOpStorageType("asin", typeof(CpuStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(asin_func, result, src); }

        private MethodInfo acos_func = NativeWrapper.GetMethod("TS_Acos");
        [RegisterOpStorageType("acos", typeof(CpuStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(acos_func, result, src); }

        private MethodInfo atan_func = NativeWrapper.GetMethod("TS_Atan");
        [RegisterOpStorageType("atan", typeof(CpuStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(atan_func, result, src); }


        private MethodInfo sinh_func = NativeWrapper.GetMethod("TS_Sinh");
        [RegisterOpStorageType("sinh", typeof(CpuStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sinh_func, result, src); }

        private MethodInfo cosh_func = NativeWrapper.GetMethod("TS_Cosh");
        [RegisterOpStorageType("cosh", typeof(CpuStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cosh_func, result, src); }

        private MethodInfo tanh_func = NativeWrapper.GetMethod("TS_Tanh");
        [RegisterOpStorageType("tanh", typeof(CpuStorage))]
        public Tensor Tanh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, src); }


        private MethodInfo sigmoid_func = NativeWrapper.GetMethod("TS_Sigmoid");
        [RegisterOpStorageType("sigmoid", typeof(CpuStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sigmoid_func, result, src); }

        private MethodInfo tanhD_func = NativeWrapper.GetMethod("TS_TanhD");
        [RegisterOpStorageType("tanhD", typeof(CpuStorage))]
        public Tensor TanhD(Tensor result, Tensor resW, Tensor resG) { return NativeWrapper.InvokeNullableResultElementwise(tanhD_func, result, resW, resG); }


        private MethodInfo add3_func = NativeWrapper.GetMethod("TS_Add3");
        [RegisterOpStorageType("add3", typeof(CpuStorage))]
        public Tensor Add3(Tensor result, Tensor x, Tensor y, Tensor z) { return NativeWrapper.InvokeNullableResultElementwise(add3_func, result, x, y, z); }

        private MethodInfo add4_func = NativeWrapper.GetMethod("TS_Add4");
        [RegisterOpStorageType("add4", typeof(CpuStorage))]
        public Tensor Add4(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return NativeWrapper.InvokeNullableResultElementwise(add4_func, result, x, y, z, w); }


        private MethodInfo addmul_func = NativeWrapper.GetMethod("TS_AddMul");
        [RegisterOpStorageType("addmul", typeof(CpuStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return NativeWrapper.InvokeNullableResultElementwise(addmul_func, result, x, y, z); }


        private MethodInfo atan2_func = NativeWrapper.GetMethod("TS_Atan2");
        [RegisterOpStorageType("atan2", typeof(CpuStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return NativeWrapper.InvokeNullableResultElementwise(atan2_func, result, srcY, srcX); }

        private MethodInfo pow_func = NativeWrapper.GetMethod("TS_Pow");
        [RegisterOpStorageType("pow", typeof(CpuStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value) { return NativeWrapper.InvokeNullableResultElementwise(pow_func, result, src, value); }

        private MethodInfo tpow_func = NativeWrapper.GetMethod("TS_Tpow");
        [RegisterOpStorageType("tpow", typeof(CpuStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tpow_func, result, value, src); }

        private MethodInfo lerp_func = NativeWrapper.GetMethod("TS_Lerp");
        [RegisterOpStorageType("lerp", typeof(CpuStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, srcA, srcB, weight); }

        private MethodInfo clamp_func = NativeWrapper.GetMethod("TS_Clamp");
        [RegisterOpStorageType("clamp", typeof(CpuStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, src, min, max); }



        private MethodInfo add_func = NativeWrapper.GetMethod("TS_Add");
        [RegisterOpStorageType("addv", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(add_func, result, lhs, rhs); }

        private MethodInfo sub_func = NativeWrapper.GetMethod("TS_Sub");
        [RegisterOpStorageType("subv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(sub_func, result, lhs, rhs); }

        private MethodInfo rsub_func = NativeWrapper.GetMethod("TS_Rsub");
        [RegisterOpStorageType("rsubv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rsub_func, result, rhs, lhs); }

        private MethodInfo mul_func = NativeWrapper.GetMethod("TS_Mul");
        [RegisterOpStorageType("mulv", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mul_func, result, lhs, rhs); }

        private MethodInfo div_func = NativeWrapper.GetMethod("TS_Div");
        [RegisterOpStorageType("divv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(div_func, result, lhs, rhs); }

        private MethodInfo rdiv_func = NativeWrapper.GetMethod("TS_Rdiv");
        [RegisterOpStorageType("rdivv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rdiv_func, result, rhs, lhs); }

        private MethodInfo mod_func = NativeWrapper.GetMethod("TS_Mod");
        [RegisterOpStorageType("modv", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mod_func, result, lhs, rhs); }


        private MethodInfo gtValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("gtValue", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtValue_func, result, lhs, rhs); }

        private MethodInfo ltValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("ltValue", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltValue_func, result, lhs, rhs); }

        private MethodInfo geValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("geValue", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(geValue_func, result, lhs, rhs); }

        private MethodInfo leValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("leValue", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(leValue_func, result, lhs, rhs); }

        private MethodInfo eqValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("eqValue", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqValue_func, result, lhs, rhs); }

        private MethodInfo neValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("neValue", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(neValue_func, result, lhs, rhs); }



        private MethodInfo cadd_func = NativeWrapper.GetMethod("TS_CAdd");
        [RegisterOpStorageType("addt", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cadd_func, result, lhs, rhs); }

        private MethodInfo csub_func = NativeWrapper.GetMethod("TS_CSub");
        [RegisterOpStorageType("subt", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(csub_func, result, lhs, rhs); }

        private MethodInfo cmul_func = NativeWrapper.GetMethod("TS_CMul");
        [RegisterOpStorageType("mult", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmul_func, result, lhs, rhs); }

        private MethodInfo cdiv_func = NativeWrapper.GetMethod("TS_CDiv");
        [RegisterOpStorageType("divt", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cdiv_func, result, lhs, rhs); }

        private MethodInfo cmod_func = NativeWrapper.GetMethod("TS_CMod");
        [RegisterOpStorageType("modt", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmod_func, result, lhs, rhs); }


        private MethodInfo gtTensor_func = NativeWrapper.GetMethod("TS_gtTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtTensor_func, result, lhs, rhs); }

        private MethodInfo ltTensor_func = NativeWrapper.GetMethod("TS_ltTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltTensor_func, result, lhs, rhs); }

        private MethodInfo geTensor_func = NativeWrapper.GetMethod("TS_geTensor");
        [RegisterOpStorageType("geTensor", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(geTensor_func, result, lhs, rhs); }

        private MethodInfo leTensor_func = NativeWrapper.GetMethod("TS_leTensor");
        [RegisterOpStorageType("leTensor", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(leTensor_func, result, lhs, rhs); }

        private MethodInfo eqTensor_func = NativeWrapper.GetMethod("TS_eqTensor");
        [RegisterOpStorageType("eqTensor", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqTensor_func, result, lhs, rhs); }

        private MethodInfo neTensor_func = NativeWrapper.GetMethod("TS_neTensor");
        [RegisterOpStorageType("neTensor", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(neTensor_func, result, lhs, rhs); }


        private MethodInfo sum_func = NativeWrapper.GetMethod("TS_Sum");
        [RegisterOpStorageType("sum", typeof(CpuStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(sum_func, result, src, dimension); }

        private MethodInfo prod_func = NativeWrapper.GetMethod("TS_Prod");
        [RegisterOpStorageType("prod", typeof(CpuStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(prod_func, result, src, dimension); }

        private MethodInfo min_func = NativeWrapper.GetMethod("TS_Min");
        [RegisterOpStorageType("min", typeof(CpuStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(min_func, result, src, dimension); }

        private MethodInfo max_func = NativeWrapper.GetMethod("TS_Max");
        [RegisterOpStorageType("max", typeof(CpuStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(max_func, result, src, dimension); }


        private MethodInfo argmin_func = NativeWrapper.GetMethod("TS_Argmin");
        [RegisterOpStorageType("argmin", typeof(CpuStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }

        private MethodInfo argmax_func = NativeWrapper.GetMethod("TS_Argmax");
        [RegisterOpStorageType("argmax", typeof(CpuStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }



        private MethodInfo mean_func = NativeWrapper.GetMethod("TS_Mean");
        [RegisterOpStorageType("mean", typeof(CpuStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(mean_func, result, src, dimension); }

        private MethodInfo norm_func = NativeWrapper.GetMethod("TS_Norm");
        [RegisterOpStorageType("norm", typeof(CpuStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return NativeWrapper.InvokeNullableResultDimensionwise(norm_func, result, src, dimension, value); }

        private MethodInfo std_func = NativeWrapper.GetMethod("TS_Std");
        [RegisterOpStorageType("std", typeof(CpuStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(std_func, result, src, dimension, normByN); }

        private MethodInfo var_func = NativeWrapper.GetMethod("TS_Var");
        [RegisterOpStorageType("var", typeof(CpuStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(var_func, result, src, dimension, normByN); }



        private MethodInfo sumall_func = NativeWrapper.GetMethod("TS_SumAll");
        [RegisterOpStorageType("sumall", typeof(CpuStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(sumall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo prodall_func = NativeWrapper.GetMethod("TS_ProdAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(prodall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo minall_func = NativeWrapper.GetMethod("TS_MinAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(minall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo maxall_func = NativeWrapper.GetMethod("TS_MaxAll");
        [RegisterOpStorageType("maxall", typeof(CpuStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(maxall_func, writeTarget, src);
            return writeTarget;
        }


        private MethodInfo meanall_func = NativeWrapper.GetMethod("TS_MeanAll");
        [RegisterOpStorageType("meanall", typeof(CpuStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(meanall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo varall_func = NativeWrapper.GetMethod("TS_VarAll");
        [RegisterOpStorageType("varall", typeof(CpuStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(varall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo stdall_func = NativeWrapper.GetMethod("TS_StdAll");
        [RegisterOpStorageType("stdall", typeof(CpuStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(stdall_func, writeTarget, src);
            return writeTarget;
        }

        private MethodInfo normall_func = NativeWrapper.GetMethod("TS_NormAll");
        [RegisterOpStorageType("normall", typeof(CpuStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(normall_func, writeTarget, src, value);
            return writeTarget;
        }
    }
}
