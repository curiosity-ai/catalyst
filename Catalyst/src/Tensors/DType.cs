using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors
{
    public enum DType
    {
        Float32 = 0,
        Float16 = 1,
        Float64 = 2,
        Int32 = 3,
        UInt8 = 4,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Half
    {
        public ushort value;
    }


    public static class DTypeExtensions
    {
        public static int Size(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return 2;
                case DType.Float32: return 4;
                case DType.Float64: return 8;
                case DType.Int32: return 4;
                case DType.UInt8: return 1;
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }

        public static Type ToCLRType(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return typeof(Half);
                case DType.Float32: return typeof(float);
                case DType.Float64: return typeof(double);
                case DType.Int32: return typeof(int);
                case DType.UInt8: return typeof(byte);
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }
    }

    public static class DTypeBuilder
    {
        public static DType FromCLRType(Type type)
        {
            if (type == typeof(Half)) return DType.Float16;
            else if (type == typeof(float)) return DType.Float32;
            else if (type == typeof(double)) return DType.Float64;
            else if (type == typeof(int)) return DType.Int32;
            else if (type == typeof(byte)) return DType.UInt8;
            else
                throw new NotSupportedException("No corresponding DType value for CLR type " + type);
        }
    }
}
