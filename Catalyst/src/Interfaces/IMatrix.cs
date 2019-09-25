using System;
using System.IO;

namespace Catalyst
{
    public interface IMatrix
    {
        int Rows { get; }
        int Columns { get; }

        float this[int i, int j] { get; set; }

        void AddToRow(ReadOnlySpan<float> vec, int i, float a);

        void AddToRow(ReadOnlySpan<float> vec, int i);

        float DotRow(ReadOnlySpan<float> vec, int i);

        float DotRow(ReadOnlySpan<float> vec, ReadOnlySpan<float> data);

        Span<float> GetRow(int row);

        Matrix Multiply(Matrix other);

        void ResizeAndFillRows(int newRows, float a);

        void ToStream(Stream stream, QuantizationType quantization);

        Matrix Transpose();

        Matrix Uniform(float a);

        Matrix Zero();
    }
}