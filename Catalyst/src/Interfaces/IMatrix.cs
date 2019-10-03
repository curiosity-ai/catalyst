using System;
using System.IO;

namespace Catalyst
{

    public interface IMatrix
    {
        int Rows { get; }
        int Columns { get; }

        float this[int i, int j] { get; set; }
#if NETCOREAPP3_0

        void AddToRow(ReadOnlySpan<float> vec, int i, float a);

        void AddToRow(ReadOnlySpan<float> vec, int i);

        float DotRow(ReadOnlySpan<float> vec, int i);

        float DotRow(ReadOnlySpan<float> vec, ReadOnlySpan<float> other);

        Span<float> GetRow(int row);
#else
        void AddToRow(float[] vec, int i, float a);

        void AddToRow(float[] vec, int i);

        float DotRow(float[] vec, int i);

        float DotRow(float[] vec, float[] other);

        float[] GetRow(int row);
#endif

        Matrix Multiply(Matrix other);

        void ResizeAndFillRows(int newRows, float a);

        void ToStream(Stream stream, QuantizationType quantization);

        Matrix Transpose();

        Matrix Uniform(float a);

        Matrix Zero();
    }
}