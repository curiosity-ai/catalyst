#if NET5_0_OR_GREATER
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
    /// <summary>
    /// Represents a matrix of floating-point numbers.
    /// </summary>
    // Cannot use [MessagePackObject] here, as it will limit the maximum size of the matrix - have to instead serialize manually to a stream
    public class Matrix : IMatrix
    {
        /// <summary>Gets the number of rows in the matrix.</summary>
        public int Rows { get; private set; }

        /// <summary>Gets the number of columns in the matrix.</summary>
        public int Columns { get; private set; }

        /// <summary>The raw matrix data.</summary>
        public float[][] Data;

        internal Matrix()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class.
        /// </summary>
        /// <param name="rows">The number of rows.</param>
        /// <param name="columns">The number of columns.</param>
        /// <param name="data">The raw data.</param>
        public Matrix(int rows, int columns, float[][] data)
        {
            Rows = rows;
            Columns = columns;
            Data = data;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class with the specified dimensions.
        /// </summary>
        /// <param name="rows">The number of rows.</param>
        /// <param name="columns">The number of columns.</param>
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

        /// <summary>
        /// Gets the specified row.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>The row data.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetRow(int row)
        {
            return Data[row];
        }

        /// <summary>
        /// Gets a reference to the specified row.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>A reference to the row data.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float[] GetRowRef(int row)
        {
            return ref Data[row];
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class with the specified raw data.
        /// </summary>
        /// <param name="data">The raw data.</param>
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

        /// <summary>
        /// Writes the matrix to a stream with the specified quantization.
        /// </summary>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="quantization">The quantization type.</param>
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

        /// <summary>
        /// Reads a matrix from a stream with the specified quantization.
        /// </summary>
        /// <param name="stream">The stream to read from.</param>
        /// <param name="quantization">The quantization type.</param>
        /// <returns>A <see cref="Matrix"/> instance.</returns>
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

        /// <summary>
        /// Gets or sets the value at the specified row and column.
        /// </summary>
        /// <param name="i">The row index.</param>
        /// <param name="j">The column index.</param>
        /// <returns>The value.</returns>
        public float this[int i, int j]
        {
            get { return Data[i][j]; }
            set { Data[i][j] = value; }
        }

        /// <summary>
        /// Gets or sets the specified row.
        /// </summary>
        /// <param name="i">The row index.</param>
        /// <returns>The row data.</returns>
        public float[] this[int i]
        {
            get { return Data[i]; }
            set { value.AsSpan().CopyTo(Data[i].AsSpan()); }
        }

        /// <summary>
        /// Fills the matrix with zeros.
        /// </summary>
        /// <returns>The current matrix instance.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Zero()
        {
            for (int i = 0; i < Rows; i++)
            {
                Data[i].Zero();
            }
            return this;
        }

        /// <summary>
        /// Fills the matrix with values from a uniform distribution between -a and a.
        /// </summary>
        /// <param name="a">The distribution parameter.</param>
        /// <returns>The current matrix instance.</returns>
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

        /// <summary>
        /// Resizes the matrix and fills new rows with values from a uniform distribution.
        /// </summary>
        /// <param name="newRows">The new number of rows.</param>
        /// <param name="a">The distribution parameter.</param>
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

        /// <summary>
        /// Adds a scaled vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        /// <param name="a">The scaling factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(float[] vec, int i, float a)
        {
            SIMD.MultiplyAndAdd(Data[i], vec, a);
        }

        /// <summary>
        /// Adds a vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(float[] vec, int i)
        {
            SIMD.Add(Data[i], vec);
        }

        /// <summary>
        /// Computes the dot product of a vector and the specified row.
        /// </summary>
        /// <param name="vec">The vector.</param>
        /// <param name="i">The row index.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(float[] vec, int i)
        {
            var d = SIMD.DotProduct(Data[i], vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        /// <summary>
        /// Computes the dot product of two vectors.
        /// </summary>
        /// <param name="vec">The first vector.</param>
        /// <param name="data">The second vector data.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(float[] vec, float[] data)
        {
            var d = SIMD.DotProduct(data, vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        /// <summary>
        /// Multiplies this matrix by another matrix.
        /// </summary>
        /// <param name="other">The other matrix.</param>
        /// <returns>The resulting matrix.</returns>
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

        /// <summary>
        /// Transposes the matrix.
        /// </summary>
        /// <returns>The transposed matrix.</returns>
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