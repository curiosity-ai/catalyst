#if NETCOREAPP3_0 || NETCOREAPP3_1 || NET5_0
#else

using MessagePackCompat;
using Mosaik.Core;
using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using System.Linq;
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

        public float[][] Data;

        internal Matrix()
        {
        }

        public Matrix(int rows, int columns, float[][] data)
        {
            Rows = rows;
            Columns = columns;
            Data = data;
        }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Data = new float[Rows][];
            for (int i = 0; i < Rows; i++)
            {
                Data[i] = new float[Columns];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetRow(int row)
        {
            return Data[row];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float[] GetRowRef(int row)
        {
            return ref Data[row];
        }

        public Matrix(float[][] data)
        {
            Rows = data.Length;
            Columns = data[0].Length;
            Data = data;
        }

        internal float[][] ToArray()
        {
            var m = new float[Rows][];
            for (int i = 0; i < Rows; i++)
            {
                m[i] = Data[i].ToArray();
            }
            return m;

        }

        public void ToStream(Stream stream, QuantizationType quantization)
        {
            MessagePackBinary.WriteInt32(stream, Rows);
            MessagePackBinary.WriteInt32(stream, Columns);

            switch (quantization)
            {
                case QuantizationType.None:
                {
                    var byteArray = new byte[Columns * sizeof(float)];
                    for (int i = 0; i < Rows; i++)
                    {
                        System.Buffer.BlockCopy(Data[i], 0, byteArray, 0, byteArray.Length);
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
                        for (int j = 0; j < Columns; j++)
                        {
                            bits.Set(j, Data[i][j] > 0);
                        }
                        bits.CopyTo(byteArray, 0);
                        System.Buffer.BlockCopy(Data[i], 0, byteArray, 0, byteArray.Length);
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
            var Rows    = MessagePackBinary.ReadInt32(stream);
            var Columns = MessagePackBinary.ReadInt32(stream);
            var m       = new Matrix(Rows, Columns);

            switch (quantization)
            {
                case QuantizationType.None:
                {
                    for (int i = 0; i < Rows; i++)
                    {
                        var byteArray = MessagePackBinary.ReadBytes(stream);
                        System.Buffer.BlockCopy(byteArray, 0, m.Data[i], 0, byteArray.Length);
                    }
                    break;
                }
                case QuantizationType.OneBit:
                {
                    for (int i = 0; i < Rows; i++)
                    {
                        var byteArray = MessagePackBinary.ReadBytes(stream);
                        var bits = new BitArray(byteArray);
                        for (int j = 0; j < Columns; j++)
                        {
                            m.Data[i][j] = bits[j] ? 0.33f : -0.33f;
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
            get { return Data[i][j]; }
            set { Data[i][j] = value; }
        }

        public float[] this[int i]
        {
            get { return Data[i]; }
            set { value.AsSpan().CopyTo(Data[i].AsSpan()); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Zero()
        {
            for (int i = 0; i < Rows; i++)
            {
                Data[i].Zero();
            }
            return this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Uniform(float a)
        {
            float a2 = 2 * a;
            float an = -a;

            Parallel.For(0, Rows, (i) =>
            {
                ThreadSafeFastRandom.NextFloats(Data[i]);
                SIMD.Multiply(Data[i], a2);
                SIMD.Add(Data[i], an);
            });
            return this;
        }

        public void ResizeAndFillRows(int newRows, float a)
        {
            Array.Resize(ref Data, newRows);
            for (int i = Rows; i < newRows; i++)
            {
                Data[i] = new float[Columns];
                for (int j = 0; j < Columns; j++)
                {
                    Data[i][j] = (float)(ThreadSafeRandom.NextDouble()) * (2 * a) - a;
                }
            }
            Rows = newRows;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(float[] vec, int i, float a)
        {
            SIMD.MultiplyAndAdd(Data[i], vec, a);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(float[] vec, int i)
        {
            SIMD.Add(Data[i], vec);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(float[] vec, int i)
        {
            var d = SIMD.DotProduct(Data[i], vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(float[] vec, float[] data)
        {
            var d = SIMD.DotProduct(data, vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        public Matrix Multiply(Matrix other)
        {
            var M = new Matrix(Rows, other.Columns);
            //var OT = other.Transpose();

            for (int i = 0; i < M.Rows; i++)
            {
                for (int j = 0; j < M.Columns; j++)
                {
                    //M.Data[i][j] = SIMD.DotProduct(ref Data[i], ref OT.Data[j]);
                    for (int k = 0; k < this.Rows; k++)
                    {
                        M.Data[i][j] += Data[i][k] * other.Data[k][j];
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
                    M.Data[j][i] = Data[i][j];
                }
            }
            return M;
        }
    }
}

#endif