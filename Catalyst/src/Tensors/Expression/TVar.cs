//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;

//namespace Catalyst.Tensors.Expression
//{
//    public class TVar
//    {
//        private TExpression expression;

//        public TVar(TExpression expression)
//        {
//            this.expression = expression;
//        }

//        // Note that this is not compatible with the implementation of operator ==
//        // This merely checks for reference equality
//        public override bool Equals(object obj)
//        {
//            return base.Equals(obj);
//        }

//        public override int GetHashCode()
//        {
//            return base.GetHashCode();
//        }


//        public TExpression Expression { get { return expression; } }

//        public static implicit operator TVar(Tensor value)
//        {
//            return new TVar(new TensorValueExpression(value));
//        }

//        public SVar ToScalar()
//        {
//            return new SVar(new DelegateScalarExpression(() =>
//            {
//                using (var result = this.Expression.Evaluate(null))
//                {
//                    return result.GetElementAsFloat(0);
//                }
//            }));
//        }


//        public static TVar Fill(SVar value, IAllocator allocator, DType type, params long[] sizes) { return new TVar(new FillExpression(allocator, type, sizes, res => Ops.Fill(res, value.Evaluate()))); }


//        public static TVar operator -(TVar src) { return new TVar(new UnaryTensorExpression(src.Expression, Ops.Neg)); }

//        public static TVar operator +(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
//        public static TVar operator +(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Add)); }
//        public static TVar operator *(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Mul)); }
//        public static TVar operator *(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Mul)); }

//        public static TVar operator -(SVar lhs, TVar rhs) { return new TVar(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
//        public static TVar operator -(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
//        public static TVar operator /(SVar lhs, TVar rhs) { return new TVar(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Div)); }
//        public static TVar operator /(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Div)); }

//        public static TVar operator +(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
//        public static TVar operator -(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }

//        public static TVar operator >(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
//        public static TVar operator <(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
//        public static TVar operator >=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
//        public static TVar operator <=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
//        public static TVar operator ==(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
//        public static TVar operator !=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }

//        public static TVar operator >(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
//        public static TVar operator <(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
//        public static TVar operator >=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
//        public static TVar operator <=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
//        public static TVar operator ==(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
//        public static TVar operator !=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }


//        // Use symmetry of these Scalar/Tensor ops to share kernels with the Tensor/Scalar versions
//        public static TVar operator >(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessThan)); }
//        public static TVar operator <(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterThan)); }
//        public static TVar operator >=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessOrEqual)); }
//        public static TVar operator <=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterOrEqual)); }
//        public static TVar operator ==(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.EqualTo)); }
//        public static TVar operator !=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.NotEqual)); }


//        public TVar Dot(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Dot)); }

//        // Returns beta * this + alpha * m1 * m2
//        public TVar Addmm(float beta, float alpha, TVar m1, TVar m2) { return new TVar(new AddmmExpression(beta, this.Expression, alpha, m1.Expression, m2.Expression)); }

//        public TVar CMul(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Mul)); }
//        public TVar CDiv(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Div)); }

//        public TVar Div(SVar rhs) { return new TVar(new BinaryTensorScalarExpression(this.Expression, rhs.Expression, Ops.Div)); }


//        public TVar Abs() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
//        public TVar Sign() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sign)); }

//        public TVar Sqrt() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
//        public TVar Exp() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Exp)); }
//        public TVar Log() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Log)); }
//        public TVar Log1p() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Log1p)); }
//        public TVar Floor() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Floor)); }
//        public TVar Ceil() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Ceil)); }
//        public TVar Round() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Round)); }
//        public TVar Trunc() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Trunc)); }
//        public TVar Frac() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Frac)); }

//        public TVar Sin() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
//        public TVar Cos() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Cos)); }
//        public TVar Tan() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Tan)); }

//        public TVar Asin() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Asin)); }
//        public TVar Acos() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Acos)); }
//        public TVar Atan() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Atan)); }

//        public TVar Sinh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sinh)); }
//        public TVar Cosh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Cosh)); }
//        public TVar Tanh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Tanh)); }

//        public TVar Sigmoid() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sigmoid)); }

//        public TVar Pow(SVar y) { return new TVar(new BinaryTensorScalarExpression(this.Expression, y.Expression, Ops.Pow)); }
//        public TVar Clamp(SVar min, SVar max) { return new TVar(new UnaryTensorExpression(this.Expression, (res, src) => Ops.Clamp(res, src, min.Evaluate(), max.Evaluate()))); }

//        public static TVar Atan2(TVar y, TVar x) { return new TVar(new BinaryTensorTensorExpression(x.Expression, y.Expression, Ops.Atan2)); }
//        public static TVar Lerp(TVar a, TVar b, SVar weight) { return new TVar(new BinaryTensorTensorExpression(a.Expression, b.Expression, (res, aVal, bVal) => Ops.Lerp(res, aVal, bVal, weight.Evaluate()))); }


//        public TVar Sum(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Sum(result, src, dimension))); }
//        public TVar Prod(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Prod(result, src, dimension))); }
//        public TVar Min(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Min(result, src, dimension))); }
//        public TVar Max(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Max(result, src, dimension))); }
//        public TVar Argmin(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmin(result, src, dimension))); }
//        public TVar Argmax(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmax(result, src, dimension))); }

//        public TVar Mean(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Mean(result, src, dimension))); }
//        public TVar Norm(int dimension, float value) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Norm(result, src, dimension, value))); }
//        public TVar Std(int dimension, bool normByN = false) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Std(result, src, dimension, normByN))); }
//        public TVar Var(int dimension, bool normByN = false) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Var(result, src, dimension, normByN))); }


//        public TVar SumAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.SumAll(result, src))); }
//        public TVar ProdAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.ProdAll(result, src))); }
//        public TVar MinAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MinAll(result, src))); }
//        public TVar MaxAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MaxAll(result, src))); }

//        public TVar MeanAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MeanAll(result, src))); }
//        public TVar VarAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.VarAll(result, src))); }
//        public TVar StdAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.StdAll(result, src))); }
//        public TVar NormAll(float value) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.NormAll(result, src, value))); }


//        public TVar Gather(int dimension, TVar indices) { return new TVar(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Gather(res, src, dimension, ind))); }
//        public TVar Scatter(int dimension, TVar indices) { return new TVar(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Scatter(res, src, dimension, ind))); }

//        // Returns a copy of this tensor, with the given indices filled with the given value.
//        // If, when this op is evaluated, the write target is the same tensor as this, then the copy is unnecessary and is skipped.
//        public TVar ScatterFill(SVar value, int dimension, TVar indices) { return new TVar(new ScatterFillExpression(this.Expression, value, dimension, indices.Expression)); }



//        public static TVar RandomUniform(SeedSource seedSource, SVar min, SVar max, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomUniform(res, seedSource, min.Evaluate(), max.Evaluate())));
//        }

//        public static TVar RandomNormal(SeedSource seedSource, SVar mean, SVar stdv, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
//        }

//        public static TVar RandomExponential(SeedSource seedSource, SVar lambda, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomExponential(res, seedSource, lambda.Evaluate())));
//        }

//        public static TVar RandomCauchy(SeedSource seedSource, SVar median, SVar sigma, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomCauchy(res, seedSource, median.Evaluate(), sigma.Evaluate())));
//        }

//        public static TVar RandomLogNormal(SeedSource seedSource, SVar mean, SVar stdv, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomLogNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
//        }

//        public static TVar RandomGeometric(SeedSource seedSource, SVar p, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomGeometric(res, seedSource, p.Evaluate())));
//        }

//        public static TVar RandomBernoulli(SeedSource seedSource, SVar p, IAllocator allocator, DType type, params long[] sizes)
//        {
//            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomBernoulli(res, seedSource, p.Evaluate())));
//        }



//        public TVar AsType(DType elementType)
//        {
//            return new TVar(new AsTypeExpression(this.Expression, elementType));
//        }

//        public TVar ToDevice(IAllocator device)
//        {
//            return new TVar(new ToDeviceExpression(this.Expression, device));
//        }
        
//        public Tensor Evaluate()
//        {
//            return expression.Evaluate(null);
//        }

//        public void Evaluate(TVar result)
//        {
//            if (!result.Expression.IsValidLvalue)
//                throw new InvalidOperationException("cannot write to given result - it is not a valid lvalue");

//            using (var res = result.Expression.Evaluate(null))
//            {
//                this.expression.Evaluate(res);
//            }
//        }

//        public static TVar FromArray(Array array, IAllocator allocator)
//        {
//            return new TVar(new FromArrayExpression(allocator, array));
//        }



//        public TVar Select(int dimension, long index) { return new TVar(new ViewExpression(this.Expression, src => src.Select(dimension, index))); }
//        public TVar Narrow(int dimension, long startIndex, long size) { return new TVar(new ViewExpression(this.Expression, src => src.Narrow(dimension, startIndex, size))); }
//        public TVar Transpose() { return new TVar(new ViewExpression(this.Expression, src => src.Transpose())); }
//        public TVar Transpose(int dim1, int dim2) { return new TVar(new ViewExpression(this.Expression, src => src.Transpose(dim1, dim2))); }
//        public TVar Permute(params int[] dims) { return new TVar(new ViewExpression(this.Expression, src => src.Permute(dims))); }
//        public TVar View(params long[] sizes) { return new TVar(new ViewExpression(this.Expression, src => src.View(sizes))); }
//        public TVar Expand(params long[] sizes) { return new TVar(new ViewExpression(this.Expression, src => src.Expand(sizes))); }
//        public TVar Squeeze() { return new TVar(new ViewExpression(this.Expression, src => src.Squeeze())); }
//    }

//    public static class TensorVarExtensions
//    {
//        public static TVar TVar(this Tensor value)
//        {
//            return new Expression.TVar(new TensorValueExpression(value));
//        }
//    }
//}
