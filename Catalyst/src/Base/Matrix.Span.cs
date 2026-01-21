#if NET5_0_OR_GREATER
using MessagePackCompat;
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
        public float[] Data;

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

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class with the specified dimensions.
        /// </summary>
        /// <param name="rows">The number of rows.</param>
        /// <param name="columns">The number of columns.</param>
        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Data = new float[rows * columns];
        }

        /// <summary>
        /// Gets the specified row.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>A span containing the row data.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Span<float> GetRow(int row)
        {
            return Data.AsSpan().Slice(row*Columns, Columns);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Matrix"/> class with the specified raw data.
        /// </summary>
        /// <param name="data">The raw data.</param>
        public Matrix(float[][] data)
        {
            Rows = data.Length;
            Columns = data[0].Length;
            for(int i = 0; i < Rows; i++)
            {
                data[i].AsSpan().CopyTo(Data.AsSpan().Slice(i*Columns, Columns));
            }
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

        /// <summary>
        /// Gets or sets the value at the specified row and column.
        /// </summary>
        /// <param name="i">The row index.</param>
        /// <param name="j">The column index.</param>
        /// <returns>The value.</returns>
        public float this[int i, int j]
        {
            get { return Data[i * Columns + j]; }
            set { Data[i * Columns + j] = value; }
        }

        /// <summary>
        /// Gets or sets the specified row.
        /// </summary>
        /// <param name="i">The row index.</param>
        /// <returns>A span containing the row data.</returns>
        public Span<float> this[int i]
        {
            get { return Data.AsSpan().Slice(i * Columns, Columns); }
            set { value.CopyTo(Data.AsSpan().Slice(i * Columns, Columns)); }
        }


        /// <summary>
        /// Fills the matrix with zeros.
        /// </summary>
        /// <returns>The current matrix instance.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Zero()
        {
            Data.AsSpan().Fill(0f);
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
            
            ThreadSafeFastRandom.NextFloats(Data.AsSpan());
            SIMD.Multiply(Data.AsSpan(), a2);
            SIMD.Add(Data.AsSpan(), an);

            return this;
        }

        /// <summary>
        /// Resizes the matrix and fills new rows with values from a uniform distribution.
        /// </summary>
        /// <param name="newRows">The new number of rows.</param>
        /// <param name="a">The distribution parameter.</param>
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

        /// <summary>
        /// Adds a scaled vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        /// <param name="a">The scaling factor.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(ReadOnlySpan<float> vec, int i, float a)
        {
            SIMD.MultiplyAndAdd(Data.AsSpan().Slice(i*Columns, Columns), vec, a);
        }

        /// <summary>
        /// Adds a vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(ReadOnlySpan<float> vec, int i)
        {
            SIMD.Add(Data.AsSpan().Slice(i * Columns, Columns), vec);
        }

        /// <summary>
        /// Computes the dot product of a vector and the specified row.
        /// </summary>
        /// <param name="vec">The vector.</param>
        /// <param name="i">The row index.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ReadOnlySpan<float> vec, int i)
        {
            var d = SIMD.DotProduct(Data.AsSpan().Slice(i * Columns, Columns), vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        /// <summary>
        /// Computes the dot product of two vectors.
        /// </summary>
        /// <param name="vec">The first vector.</param>
        /// <param name="other">The second vector.</param>
        /// <returns>The dot product.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ReadOnlySpan<float> vec, ReadOnlySpan<float> other)
        {
            var d = SIMD.DotProduct(vec, other);
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
                    M.Data[j * Columns + i] = Data[i * Columns + j];
                }
            }
            return M;
        }
    }
}
#endif