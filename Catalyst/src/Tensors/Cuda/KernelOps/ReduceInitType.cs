using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.KernelOps
{
    public enum ReduceInitType
    {
        GivenValue,
        MinValue,
        MaxValue,
    }

    public static class ReduceInitConverter
    {
        public static object GetInitValue(float value, ReduceInitType initType, DType elementType)
        {
            switch (initType)
            {
                case ReduceInitType.GivenValue: return FloatAsType(value, elementType);
                case ReduceInitType.MinValue: return GetMinValue(elementType);
                case ReduceInitType.MaxValue: return GetMaxValue(elementType);
                default:
                    throw new NotSupportedException();
            }
        }

        private static object FloatAsType(float value, DType elementType)
        {
            if (elementType == DType.Float32) return value;
            else if (elementType == DType.Float64) return (double)value;
            else if (elementType == DType.Int32) return (int)value;
            else if (elementType == DType.UInt8) return (byte)value;
            else
                throw new NotSupportedException("casting value to type " + elementType + " not supported");
        }

        private static object GetMinValue(DType elementType)
        {
            if (elementType == DType.Float32) return float.MinValue;
            else if (elementType == DType.Float64) return double.MinValue;
            else if (elementType == DType.Int32) return int.MinValue;
            else if (elementType == DType.UInt8) return byte.MinValue;
            else
                throw new NotSupportedException("getting min value of type " + elementType + " not supported");
        }

        private static object GetMaxValue(DType elementType)
        {
            if (elementType == DType.Float32) return float.MaxValue;
            else if (elementType == DType.Float64) return double.MaxValue;
            else if (elementType == DType.Int32) return int.MaxValue;
            else if (elementType == DType.UInt8) return byte.MaxValue;
            else
                throw new NotSupportedException("getting max value of type " + elementType + " not supported");
        }
    }
}
