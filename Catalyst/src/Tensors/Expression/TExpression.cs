using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.Expression
{
    public abstract class TExpression
    {
        public bool IsValidLvalue { get; private set; }

        public TExpression(bool isValidLvalue = false)
        {
            IsValidLvalue = isValidLvalue;
        }

        public abstract Tensor Evaluate(Tensor writeTarget);
    }

    public class ViewExpression : TExpression
    {
        private readonly TExpression src;
        private readonly Func<Tensor, Tensor> evaluate;

        public ViewExpression(TExpression src, Func<Tensor, Tensor> evaluate)
            : base(src.IsValidLvalue)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget != null) throw new InvalidOperationException("Cannot Select directly into another tensor");

            using (var s = src.Evaluate(null))
            {
                return evaluate(s);
            }
        }
    }

    public class FromArrayExpression : TExpression
    {
        private readonly IAllocator allocator;
        private readonly Array array;

        public FromArrayExpression(IAllocator allocator, Array array)
            : base(false)
        {
            this.allocator = allocator;
            this.array = array;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget != null)
            {
                writeTarget.CopyFrom(array);
                return writeTarget;
            }
            else
            {
                return Tensor.FromArray(allocator, array);
            }
        }
    }

    public class AsTypeExpression : TExpression
    {
        private readonly TExpression src;
        private readonly DType type;

        public AsTypeExpression(TExpression src, DType type)
        {
            this.src = src;
            this.type = type;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var srcVal = src.Evaluate(null))
            {
                if (writeTarget == null)
                {
                    writeTarget = new Tensor(srcVal.Allocator, type, srcVal.Sizes);
                }

                Ops.Copy(writeTarget, srcVal);
                return writeTarget;
            }
        }
    }

    public class ToDeviceExpression : TExpression
    {
        private readonly TExpression src;
        private readonly IAllocator allocator;

        public ToDeviceExpression(TExpression src, IAllocator allocator)
        {
            this.src = src;
            this.allocator = allocator;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var srcVal = src.Evaluate(null))
            {
                if (writeTarget == null)
                {
                    writeTarget = new Tensor(allocator, srcVal.ElementType, srcVal.Sizes);
                }

                Ops.Copy(writeTarget, srcVal);
                return writeTarget;
            }
        }
    }

    public class ScatterFillExpression : TExpression
    {
        private readonly TExpression src;
        private readonly TExpression indices;
        private readonly SVar value;
        private readonly int dimension;


        public ScatterFillExpression(TExpression src, SVar value, int dimension, TExpression indices)
        {
            this.src = src;
            this.value = value;
            this.dimension = dimension;
            this.indices = indices;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var s = src.Evaluate(null))
            using (var i = indices.Evaluate(null))
            {
                if (!writeTarget.Equals(s))
                {
                    Ops.Copy(writeTarget, s);
                }
                Ops.ScatterFill(writeTarget, value.Evaluate(), dimension, i);
            }

            return writeTarget;
        }
    }

    public class FillExpression : TExpression
    {
        private readonly IAllocator allocator;
        private readonly DType elementType;
        private readonly long[] sizes;
        private readonly Action<Tensor> fillAction;


        public FillExpression(IAllocator allocator, DType elementType, long[] sizes, Action<Tensor> fillAction)
        {
            this.allocator = allocator;
            this.elementType = elementType;
            this.sizes = sizes;
            this.fillAction = fillAction;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget == null)
                writeTarget = new Tensor(allocator, elementType, sizes);

            fillAction(writeTarget);

            return writeTarget;
        }
    }

    public class AddmmExpression : TExpression
    {
        private readonly TExpression src, m1, m2;
        private readonly float alpha, beta;

        public AddmmExpression(float beta, TExpression src, float alpha, TExpression m1, TExpression m2)
        {
            this.beta = beta;
            this.src = src;
            this.alpha = alpha;
            this.m1 = m1;
            this.m2 = m2;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var s = src.Evaluate(null))
            using (var m1Val = m1.Evaluate(null))
            using (var m2Val = m2.Evaluate(null))
            {
                return Ops.Addmm(writeTarget, beta, s, alpha, m1Val, m2Val);
            }
        }
    }


    public class TensorValueExpression : TExpression
    {
        private readonly Tensor value;

        public TensorValueExpression(Tensor value)
            : base(true)
        {
            this.value = value;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            if (writeTarget == null)
                return value.CopyRef();
            else
            {
                Ops.Copy(writeTarget, value);
                return writeTarget;
            }
        }
    }

    public class BinaryTensorTensorExpression : TExpression
    {
        private readonly TExpression left, right;
        private readonly Func<Tensor, Tensor, Tensor, Tensor> evaluate;

        public BinaryTensorTensorExpression(TExpression left, TExpression right, Func<Tensor, Tensor, Tensor, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var lhs = left.Evaluate(null))
            using (var rhs = right.Evaluate(null))
            {
                return evaluate(writeTarget, lhs, rhs);
            }
        }
    }

    public class UnaryTensorExpression : TExpression
    {
        private readonly TExpression src;
        private readonly Func<Tensor, Tensor, Tensor> evaluate;

        public UnaryTensorExpression(TExpression src, Func<Tensor, Tensor, Tensor> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var s = src.Evaluate(null))
            {
                return evaluate(writeTarget, s);
            }
        }
    }

    public class BinaryScalarTensorExpression : TExpression
    {
        public readonly SExpression left;
        public readonly TExpression right;
        public readonly Func<Tensor, float, Tensor, Tensor> evaluate;

        public BinaryScalarTensorExpression(SExpression left, TExpression right, Func<Tensor, float, Tensor, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var rhs = right.Evaluate(null))
            {
                return evaluate(writeTarget, left.Evaluate(), rhs);
            }
        }
    }

    public class BinaryTensorScalarExpression : TExpression
    {
        public readonly TExpression left;
        public readonly SExpression right;
        public readonly Func<Tensor, Tensor, float, Tensor> evaluate;

        public BinaryTensorScalarExpression(TExpression left, SExpression right, Func<Tensor, Tensor, float, Tensor> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        public override Tensor Evaluate(Tensor writeTarget)
        {
            using (var lhs = left.Evaluate(null))
            {
                return evaluate(writeTarget, lhs, right.Evaluate());
            }
        }
    }
}
