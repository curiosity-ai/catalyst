// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System.IO;

namespace Catalyst
{
    public interface IMatrix
    {
        int Rows { get; }
        int Columns { get; }

        float this[int i, int j] { get; set; }

        void AddToRow(float[] vec, int i, float a);

        void AddToRow(ref float[] vec, int i);

        float DotRow(ref float[] vec, int i);

        float DotRow(ref float[] vec, ref float[] data);

        ref float[] GetRowRef(int row);

        float[] GetRowCopy(int row);

        Matrix Multiply(Matrix other);

        void ResizeAndFillRows(int newRows, float a);

        void ToStream(Stream stream, QuantizationType quantization);

        Matrix Transpose();

        Matrix Uniform(float a);

        Matrix Zero();
    }
}