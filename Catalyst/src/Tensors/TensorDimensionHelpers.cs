using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors
{
    public static class TensorDimensionHelpers
    {
        public static long ElementCount(long[] sizes)
        {
            if (sizes.Length == 0)
                return 0;

            var total = 1L;
            for (int i = 0; i < sizes.Length; ++i)
                total *= sizes[i];
            return total;
        }

        public static long GetStorageSize(long[] sizes, long[] strides)
        {
            long offset = 0;
            for (int i = 0; i < sizes.Length; ++i)
            {
                offset += (sizes[i] - 1) * strides[i];
            }
            return offset + 1; // +1 to count last element, which is at *index* equal to offset
        }

        // Returns the stride required for a tensor to be contiguous.
        // If a tensor is contiguous, then the elements in the last dimension are contiguous in memory,
        // with lower numbered dimensions having increasingly large strides.
        public static long[] GetContiguousStride(long[] dims)
        {
            long acc = 1;
            var stride = new long[dims.Length];

            for (int i = dims.Length - 1; i >= 0; --i)
            {
                stride[i] = acc;
                acc *= dims[i];
            }

            //if (dims[dims.Length - 1] == 1)
            //{
            //    stride[dims.Length - 1] = 0;
            //}

            return stride;
        }
    }
}
