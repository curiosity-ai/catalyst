using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.ML.Data;

namespace Catalyst.Models
{
    internal delegate bool InPredicate<T>(in T value);
    internal static class VBufferUtils
    {
        /// <summary>
        /// Updates the logical length and number of physical values to be represented in
        /// <paramref name="dst"/>, while preserving the underlying buffers.
        /// </summary>
        public static void Resize<T>(ref VBuffer<T> dst, int newLogicalLength, int? valuesCount = null)
        {
            dst = VBufferEditor.Create(ref dst, newLogicalLength, valuesCount)
                .Commit();
        }
    }
}
