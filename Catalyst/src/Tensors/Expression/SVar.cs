using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.Expression
{
    public class SVar
    {
        private SExpression expression;


        public SVar(SExpression expression)
        {
            this.expression = expression;
        }


        public float Evaluate()
        {
            return expression.Evaluate();
        }

        public SExpression Expression { get { return expression; } }


        public static implicit operator SVar(float value) { return new SVar(new ConstScalarExpression(value)); }

        public static SVar operator -(SVar src) { return new SVar(new UnaryScalarExpression(src.expression, val => -val)); }

        public static SVar operator +(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l + r)); }
        public static SVar operator -(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l - r)); }
        public static SVar operator *(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l * r)); }
        public static SVar operator /(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l / r)); }
        public static SVar operator %(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l % r)); }


        public SVar Abs() { return new SVar(new UnaryScalarExpression(this.expression, val => Math.Abs(val))); }
        public SVar Sign() { return new SVar(new UnaryScalarExpression(this.expression, val => Math.Sign(val))); }

        public SVar Sqrt() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sqrt(val))); }
        public SVar Exp() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Exp(val))); }
        public SVar Log() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Log(val))); }
        public SVar Floor() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Floor(val))); }
        public SVar Ceil() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Ceiling(val))); }
        public SVar Round() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Round(val))); }
        public SVar Trunc() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Truncate(val))); }


        public SVar Sin() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sin(val))); }
        public SVar Cos() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Cos(val))); }
        public SVar Tan() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Tan(val))); }

        public SVar Asin() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Asin(val))); }
        public SVar Acos() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Acos(val))); }
        public SVar Atan() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Atan(val))); }

        public SVar Sinh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sinh(val))); }
        public SVar Cosh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Cosh(val))); }
        public SVar Tanh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Tanh(val))); }


        public SVar Pow(SVar y) { return new SVar(new BinaryScalarExpression(this.expression, y.expression, (xVal, yVal) => (float)Math.Pow(xVal, yVal))); }
        public SVar Clamp(SVar min, SVar max) { return new SVar(new DelegateScalarExpression(() => ClampFloat(this.expression.Evaluate(), min.expression.Evaluate(), max.expression.Evaluate()))); }

       // public TVar Pow(TVar y) { return new TVar(new BinaryScalarTensorExpression(this.Expression, y.Expression, Ops.Tpow)); }


        public static SVar Atan2(SVar y, SVar x) { return new SVar(new DelegateScalarExpression(() => (float)Math.Atan2(y.Evaluate(), x.Evaluate()))); }
        public static SVar Lerp(SVar a, SVar b, SVar weight) { return new SVar(new DelegateScalarExpression(() => (float)LerpFloat(a.Evaluate(), b.Evaluate(), weight.Evaluate()))); }


        private static float LerpFloat(float a, float b, float weight)
        {
            return a + weight * (b - a);
        }

        private static float ClampFloat(float value, float min, float max)
        {
            if (value < min)
                return min;
            else if (value > max)
                return max;
            else
                return value;
        }
    }
}
