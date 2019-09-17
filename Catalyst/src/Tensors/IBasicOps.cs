using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors
{
    /*
    public interface IBasicOps
    {
        Tensor NewContiguous(Tensor src);

        Tensor AsContiguous(Tensor src);

        Tensor Dot(Tensor result, Tensor lhs, Tensor rhs);

        // result = (alpha * m1 * m2) + (beta * src)
        Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2);

        //
        // Element-wise operators
        //

        void Copy(Tensor result, Tensor src);
        void Fill(Tensor result, float value);

        Tensor Concat(Tensor result, int dimension, params Tensor[] inputs);

        Tensor Abs(Tensor result, Tensor src);
        Tensor Neg(Tensor result, Tensor src);
        Tensor Sign(Tensor result, Tensor src);

        Tensor Sqrt(Tensor result, Tensor src);
        Tensor Exp(Tensor result, Tensor src);
        Tensor Log(Tensor result, Tensor src);
        Tensor Log1p(Tensor result, Tensor src);
        Tensor Floor(Tensor result, Tensor src);
        Tensor Ceil(Tensor result, Tensor src);
        Tensor Round(Tensor result, Tensor src);
        Tensor Trunc(Tensor result, Tensor src);
        Tensor Frac(Tensor result, Tensor src);

        Tensor Sin(Tensor result, Tensor src);
        Tensor Cos(Tensor result, Tensor src);
        Tensor Tan(Tensor result, Tensor src);
        Tensor Asin(Tensor result, Tensor src);
        Tensor Acos(Tensor result, Tensor src);
        Tensor Atan(Tensor result, Tensor src);
        Tensor Sinh(Tensor result, Tensor src);
        Tensor Cosh(Tensor result, Tensor src);
        Tensor Tanh(Tensor result, Tensor src);

        Tensor Sigmoid(Tensor result, Tensor src);

        Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX);
        Tensor Pow(Tensor result, Tensor src, float value);
        Tensor Tpow(Tensor result, float value, Tensor src);
        Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight);
        Tensor Clamp(Tensor result, Tensor src, float min, float max);

        Tensor Add(Tensor result, Tensor rhs, float lhs);
        Tensor Sub(Tensor result, Tensor rhs, float lhs);
        Tensor Sub(Tensor result, float rhs, Tensor lhs);
        Tensor Mul(Tensor result, Tensor rhs, float lhs);
        Tensor Div(Tensor result, Tensor rhs, float lhs);
        Tensor Div(Tensor result, float rhs, Tensor lhs);
        Tensor Mod(Tensor result, Tensor rhs, float lhs);

        Tensor GreaterThan(Tensor result, Tensor rhs, float lhs);
        Tensor LessThan(Tensor result, Tensor rhs, float lhs);
        Tensor GreaterOrEqual(Tensor result, Tensor rhs, float lhs);
        Tensor LessOrEqual(Tensor result, Tensor rhs, float lhs);
        Tensor EqualTo(Tensor result, Tensor rhs, float lhs);
        Tensor NotEqual(Tensor result, Tensor rhs, float lhs);


        Tensor Add(Tensor result, Tensor rhs, Tensor lhs);
        Tensor Sub(Tensor result, Tensor rhs, Tensor lhs);
        Tensor Mul(Tensor result, Tensor rhs, Tensor lhs);
        Tensor Div(Tensor result, Tensor rhs, Tensor lhs);
        Tensor Mod(Tensor result, Tensor rhs, Tensor lhs);

        Tensor GreaterThan(Tensor result, Tensor rhs, Tensor lhs);
        Tensor LessThan(Tensor result, Tensor rhs, Tensor lhs);
        Tensor GreaterOrEqual(Tensor result, Tensor rhs, Tensor lhs);
        Tensor LessOrEqual(Tensor result, Tensor rhs, Tensor lhs);
        Tensor EqualTo(Tensor result, Tensor rhs, Tensor lhs);
        Tensor NotEqual(Tensor result, Tensor rhs, Tensor lhs);


        //
        // Dimension-wise operators
        //

        Tensor Sum(Tensor result, Tensor src, int dimension);
        Tensor Prod(Tensor result, Tensor src, int dimension);
        Tensor Min(Tensor result, Tensor src, int dimension);
        Tensor Max(Tensor result, Tensor src, int dimension);

        Tensor Argmax(Tensor result, Tensor src, int dimension);


        Tensor Mean(Tensor result, Tensor src, int dimension);
        Tensor Norm(Tensor result, Tensor src, int dimension, float value);
        Tensor Std(Tensor result, Tensor src, int dimension, bool normByN);
        Tensor Var(Tensor result, Tensor src, int dimension, bool normByN);

        //
        // Full-tensor operators
        //

        Tensor SumAll(Tensor result, Tensor src);
        Tensor ProdAll(Tensor result, Tensor src);
        Tensor MinAll(Tensor result, Tensor src);
        Tensor MaxAll(Tensor result, Tensor src);

        Tensor MeanAll(Tensor result, Tensor src);
        Tensor VarAll(Tensor result, Tensor src);
        Tensor StdAll(Tensor result, Tensor src);
        Tensor NormAll(Tensor result, Tensor src, float value);

        float SumAll(Tensor src);
        float ProdAll(Tensor src);
        float MinAll(Tensor src);
        float MaxAll(Tensor src);

        float MeanAll(Tensor src);
        float VarAll(Tensor src);
        float StdAll(Tensor src);
        float NormAll(Tensor src, float value);
    }*/
}
