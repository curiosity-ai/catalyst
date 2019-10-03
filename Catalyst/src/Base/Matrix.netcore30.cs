#if NETCOREAPP3_0
using MessagePack;
using Mosaik.Core;
using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Catalyst
{
    // Cannot use [MessagePackObject] here, as it will limit the maximum size of the matrix - have to instead serialize manually to a stream
    public class Matrix : IMatrix
    {
        public int Rows { get; private set; }
        public int Columns { get; private set; }

        public float[] Data;

        internal Matrix()
        {
        }

        public Matrix(int rows, int columns, float[][] data)
        {
            Rows = rows;
            Columns = columns;
            for (int i = 0; i < Rows; i++)
            {
                data[i].AsSpan().CopyTo(Data.AsSpan().Slice(i * Columns, Columns));
            }
        }

        internal float[][] ToArray()
        {
            var m = new float[Rows][];
            for (int i = 0; i < Rows; i++)
            {
                m[i] = Data.AsSpan().Slice(i * Columns, Columns).ToArray();
            }
            return m;
        }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Data = new float[rows * columns];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<float> GetRow(int row)
        {
            return Data.AsSpan().Slice(row*Columns, Columns);
        }

        public Matrix(float[][] data)
        {
            Rows = data.Length;
            Columns = data[0].Length;
            for(int i = 0; i < Rows; i++)
            {
                data[i].AsSpan().CopyTo(Data.AsSpan().Slice(i*Columns, Columns));
            }
        }

        public void ToStream(Stream stream, QuantizationType quantization)
        {
            MessagePackBinary.WriteInt32(stream, Rows);
            MessagePackBinary.WriteInt32(stream, Columns);

            var data = Data.AsSpan();
            switch (quantization)
            {
                case QuantizationType.None:
                {
                    var byteArray = new byte[Columns * sizeof(float)];
                    for (int i = 0; i < Rows; i++)
                    {
                        var row = data.Slice(i * Columns, Columns);
                        MemoryMarshal.Cast<float, byte>(row).CopyTo(byteArray);
                        MessagePackBinary.WriteBytes(stream, byteArray);
                    }
                    break;
                }
                case QuantizationType.OneBit:
                {
                    var bits = new BitArray(Columns);

                    byte[] byteArray = new byte[(bits.Length - 1) / 8 + 1];
                    for (int i = 0; i < Rows; i++)
                    {
                        var row = data.Slice(i * Columns, Columns);
                        for (int j = 0; j < Columns; j++)
                        {
                            bits.Set(j, row[j] > 0);
                        }
                        bits.CopyTo(byteArray, 0);
                        MessagePackBinary.WriteBytes(stream, byteArray);
                    }
                    break;
                }
                default: throw new NotImplementedException();
            }

            stream.Flush();
        }

        public static Matrix FromStream(Stream stream, QuantizationType quantization)
        {
            var Rows = MessagePackBinary.ReadInt32(stream);
            var Columns = MessagePackBinary.ReadInt32(stream);
            var m = new Matrix(Rows, Columns);
            
            var data = m.Data.AsSpan();

            switch (quantization)
            {
                case QuantizationType.None:
                {
                    for (int i = 0; i < Rows; i++)
                    {
                        var row = data.Slice(i * Columns, Columns);
                        var byteArray = MessagePackBinary.ReadBytes(stream);
                        MemoryMarshal.Cast<byte, float>(byteArray.AsSpan()).CopyTo(row);
                    }
                    break;
                }
                case QuantizationType.OneBit:
                {
                    for (int i = 0; i < Rows; i++)
                    {
                        var byteArray = MessagePackBinary.ReadBytes(stream);
                        var bits = new BitArray(byteArray);
                        var row = data.Slice(i * Columns, Columns);
                        for (int j = 0; j < Columns; j++)
                        {
                            row[j] = bits[j] ? 0.33f : -0.33f;
                        }
                    }
                    break;
                }
                default: throw new NotImplementedException();
            }

            return m;
        }

        public float this[int i, int j]
        {
            get { return Data[i * Columns + j]; }
            set { Data[i * Columns + j] = value; }
        }

        public Span<float> this[int i]
        {
            get { return Data.AsSpan().Slice(i * Columns, Columns); }
            set { value.CopyTo(Data.AsSpan().Slice(i * Columns, Columns)); }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Zero()
        {
            Data.AsSpan().Fill(0f);
            return this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Uniform(float a)
        {
            float a2 = 2 * a;
            float an = -a;
            
            ThreadSafeFastRandom.NextFloats(Data.AsSpan());
            SIMD.Multiply(Data.AsSpan(), a2);
            SIMD.Add(Data.AsSpan(), an);

            return this;
        }

        public void ResizeAndFillRows(int newRows, float a)
        {
            float a2 = 2 * a;
            float an = -a;

            Array.Resize(ref Data, newRows * Columns);
            var toFill = Data.AsSpan().Slice(Rows * Columns);
            ThreadSafeFastRandom.NextFloats(toFill);
            SIMD.Multiply(toFill, a2);
            SIMD.Add(toFill, an);

            Rows = newRows;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(ReadOnlySpan<float> vec, int i, float a)
        {
            SIMD.MultiplyAndAdd(Data.AsSpan().Slice(i*Columns, Columns), vec, a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(ReadOnlySpan<float> vec, int i)
        {
            SIMD.Add(Data.AsSpan().Slice(i * Columns, Columns), vec);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ReadOnlySpan<float> vec, int i)
        {
            var d = SIMD.DotProduct(Data.AsSpan().Slice(i * Columns, Columns), vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ReadOnlySpan<float> vec, ReadOnlySpan<float> other)
        {
            var d = SIMD.DotProduct(vec, other);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        public Matrix Multiply(Matrix other)
        {
            var M = new Matrix(Rows, other.Columns);

            for (int i = 0; i < M.Rows; i++)
            {
                var iC = i * Columns;
                for (int j = 0; j < M.Columns; j++)
                {
                    for (int k = 0; k < Rows; k++)
                    {
                        M.Data[iC + j] += Data[iC + k] * other.Data[k*other.Columns + j];
                    }
                }
            }
            return M;
        }

        public Matrix Transpose()
        {
            var M = new Matrix(Columns, Rows);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    M.Data[j * Columns + i] = Data[i * Columns + j];
                }
            }
            return M;
        }
    }
}
#endif