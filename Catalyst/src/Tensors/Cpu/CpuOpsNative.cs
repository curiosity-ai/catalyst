using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    public enum CpuDType : int
    {
        Float32 = 0,
        Float16 = 1,
        Float64 = 2,
        Int32 = 3,
        UInt8 = 4,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct TensorRef64
    {
        public IntPtr buffer;
        public IntPtr sizes;
        public IntPtr strides;
        public int dimCount;
        public CpuDType elementType;
    }


    public static class CpuOpsNative
    {
        private const string dll = "CpuOps.dll";
        private const CallingConvention cc = CallingConvention.Cdecl;

        [DllImport(dll, CallingConvention = cc)]
        public static extern IntPtr TS_GetLastError();

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Fill(IntPtr result, float value);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Copy(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Abs(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Neg(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sign(IntPtr result, IntPtr src);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sqrt(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rsqrt(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Exp(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Log(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Log1p(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Floor(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Ceil(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Round(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Trunc(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Frac(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sin(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cos(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tan(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Asin(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Acos(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sinh(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cosh(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tanh(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sigmoid(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_TanhD(IntPtr result, IntPtr resW, IntPtr resG);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add3(IntPtr result, IntPtr x, IntPtr y, IntPtr z);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add4(IntPtr result, IntPtr x, IntPtr y, IntPtr z, IntPtr w);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_AddMul(IntPtr result, IntPtr x, IntPtr y, IntPtr z);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan2(IntPtr result, IntPtr srcY, IntPtr srcX);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Pow(IntPtr result, IntPtr src, float value);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tpow(IntPtr result, float value, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Lerp(IntPtr result, IntPtr srcA, IntPtr srcB, float weight);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Clamp(IntPtr result, IntPtr src, float min, float max);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sub(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rsub(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mul(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Div(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rdiv(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mod(IntPtr result, IntPtr lhs, float rhs);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtValue(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltValue(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geValue(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leValue(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqValue(IntPtr result, IntPtr lhs, float rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neValue(IntPtr result, IntPtr lhs, float rhs);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CAdd(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CSub(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CMul(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CDiv(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CMod(IntPtr result, IntPtr lhs, IntPtr rhs);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neTensor(IntPtr result, IntPtr lhs, IntPtr rhs);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sum(IntPtr result, IntPtr src, int dimension);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Prod(IntPtr result, IntPtr src, int dimension);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Min(IntPtr result, IntPtr src, int dimension);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Max(IntPtr result, IntPtr src, int dimension);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Argmin(IntPtr result, IntPtr src, int dimension);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Argmax(IntPtr result, IntPtr src, int dimension);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mean(IntPtr result, IntPtr src, int dimension);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Norm(IntPtr result, IntPtr src, int dimension, float value);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Std(IntPtr result, IntPtr src, int dimension, bool normByN);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Var(IntPtr result, IntPtr src, int dimension, bool normByN);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SumAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ProdAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MinAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MaxAll(IntPtr result, IntPtr src);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MeanAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_VarAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_StdAll(IntPtr result, IntPtr src);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NormAll(IntPtr result, IntPtr src, float value);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NewRNG(out IntPtr rng);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_DeleteRNG(IntPtr rng);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SetRNGSeed(IntPtr rng, int newSeed);

        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomUniform(IntPtr rng, IntPtr result, float min, float max);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomExponential(IntPtr rng, IntPtr result, float lambda);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomCauchy(IntPtr rng, IntPtr result, float median, float sigma);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomLogNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomGeometric(IntPtr rng, IntPtr result, float p);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomBernoulli(IntPtr rng, IntPtr result, float p);


        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Acc(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Copy(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);


        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateOutput_frame(IntPtr input_p, IntPtr output_p, IntPtr ind_p, long nslices, long iwidth, long iheight, long owidth, long oheight, int kW, int kH, int dW, int dH, int padW, int padH);

        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateGradInput_frame(IntPtr gradInput, IntPtr gradOutput, IntPtr ind, long nslices, long iwidth, long iheight, long owidth, long oheight, int dW, int dH);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Gather(IntPtr result, IntPtr src, int dim, IntPtr indices);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Scatter(IntPtr result, IntPtr src, int dim, IntPtr indices);
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ScatterFill(IntPtr result, float value, int dim, IntPtr indices);

    }
}
