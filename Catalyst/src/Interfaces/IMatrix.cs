using System;
using System.IO;

namespace Catalyst
{

    /// <summary>
    /// Defines an interface for a matrix.
    /// </summary>
    public interface IMatrix
    {
        /// <summary>
        /// Gets the number of rows in the matrix.
        /// </summary>
        int Rows { get; }

        /// <summary>
        /// Gets the number of columns in the matrix.
        /// </summary>
        int Columns { get; }

        /// <summary>
        /// Gets or sets the value at the specified row and column.
        /// </summary>
        /// <param name="i">The row index.</param>
        /// <param name="j">The column index.</param>
        /// <returns>The value at the specified position.</returns>
        float this[int i, int j] { get; set; }

#if  NET5_0_OR_GREATER
        /// <summary>
        /// Adds a vector to the specified row, scaled by a factor.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        /// <param name="a">The scaling factor.</param>
        void AddToRow(ReadOnlySpan<float> vec, int i, float a);

        /// <summary>
        /// Adds a vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        void AddToRow(ReadOnlySpan<float> vec, int i);

        /// <summary>
        /// Computes the dot product of a vector and the specified row.
        /// </summary>
        /// <param name="vec">The vector.</param>
        /// <param name="i">The row index.</param>
        /// <returns>The dot product.</returns>
        float DotRow(ReadOnlySpan<float> vec, int i);

        /// <summary>
        /// Computes the dot product of two vectors.
        /// </summary>
        /// <param name="vec">The first vector.</param>
        /// <param name="other">The second vector.</param>
        /// <returns>The dot product.</returns>
        float DotRow(ReadOnlySpan<float> vec, ReadOnlySpan<float> other);

        /// <summary>
        /// Gets the specified row as a span.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>A span representing the row.</returns>
        Span<float> GetRow(int row);
#else
        /// <summary>
        /// Adds a vector to the specified row, scaled by a factor.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        /// <param name="a">The scaling factor.</param>
        void AddToRow(float[] vec, int i, float a);

        /// <summary>
        /// Adds a vector to the specified row.
        /// </summary>
        /// <param name="vec">The vector to add.</param>
        /// <param name="i">The row index.</param>
        void AddToRow(float[] vec, int i);

        /// <summary>
        /// Computes the dot product of a vector and the specified row.
        /// </summary>
        /// <param name="vec">The vector.</param>
        /// <param name="i">The row index.</param>
        /// <returns>The dot product.</returns>
        float DotRow(float[] vec, int i);

        /// <summary>
        /// Computes the dot product of two vectors.
        /// </summary>
        /// <param name="vec">The first vector.</param>
        /// <param name="other">The second vector.</param>
        /// <returns>The dot product.</returns>
        float DotRow(float[] vec, float[] other);

        /// <summary>
        /// Gets the specified row as an array.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <returns>An array representing the row.</returns>
        float[] GetRow(int row);
#endif

        /// <summary>
        /// Multiplies this matrix by another matrix.
        /// </summary>
        /// <param name="other">The other matrix.</param>
        /// <returns>The result of the multiplication.</returns>
        Matrix Multiply(Matrix other);

        /// <summary>
        /// Resizes the matrix and fills new rows with a specified value.
        /// </summary>
        /// <param name="newRows">The new number of rows.</param>
        /// <param name="a">The value to fill with.</param>
        void ResizeAndFillRows(int newRows, float a);

        /// <summary>
        /// Writes the matrix to a stream with the specified quantization.
        /// </summary>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="quantization">The quantization type.</param>
        void ToStream(Stream stream, QuantizationType quantization);

        /// <summary>
        /// Transposes the matrix.
        /// </summary>
        /// <returns>The transposed matrix.</returns>
        Matrix Transpose();

        /// <summary>
        /// Fills the matrix with values from a uniform distribution.
        /// </summary>
        /// <param name="a">The distribution parameter.</param>
        /// <returns>The updated matrix.</returns>
        Matrix Uniform(float a);

        /// <summary>
        /// Fills the matrix with zeros.
        /// </summary>
        /// <returns>The updated matrix.</returns>
        Matrix Zero();
    }
}