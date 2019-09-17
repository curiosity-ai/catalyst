using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors
{
    public static class TensorSerialization
    {
        public static void Serialize(Tensor tensor, Stream stream)
        {
            using (var src = Ops.AsContiguous(tensor))
            {
                // Note: don't dispose writer - it does not own the stream's lifetime
                var writer = new System.IO.BinaryWriter(stream);

                // Can infer strides - src is contiguous
                writer.Write(tensor.DimensionCount); // int32
                writer.Write((int)tensor.ElementType);
                for (int i = 0; i < tensor.DimensionCount; ++i)
                {
                    writer.Write(tensor.Sizes[i]);
                }

                var byteCount = src.ElementType.Size() * tensor.ElementCount();
                writer.Write(byteCount);
                WriteBytes(writer, src.Storage, src.StorageOffset, byteCount);

                writer.Flush();
            }
        }

        public static Tensor Deserialize(IAllocator allocator, Stream stream)
        {
            // Note: don't dispose reader - it does not own the stream's lifetime
            var reader = new BinaryReader(stream);

            var dimCount = reader.ReadInt32();
            var elementType = (DType)reader.ReadInt32();
            var sizes = new long[dimCount];
            for (int i = 0; i < dimCount; ++i)
            {
                sizes[i] = reader.ReadInt64();
            }

            var byteCount = reader.ReadInt64();
            var result = new Tensor(allocator, elementType, sizes);

            ReadBytes(reader, result.Storage, result.StorageOffset, byteCount);

            return result;
        }

        private static void WriteBytes(BinaryWriter writer, Storage storage, long startIndex, long byteCount)
        {
            var buffer = new byte[4096];
            var bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    var length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    storage.CopyFromStorage(bufferHandle.AddrOfPinnedObject(), curStart, length);
                    writer.Write(buffer, 0, length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }

        private static void ReadBytes(BinaryReader reader, Storage storage, long startIndex, long byteCount)
        {
            var buffer = new byte[4096];
            var bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            try
            {
                long curStart = startIndex;
                long afterLastByte = startIndex + byteCount;
                while (curStart < afterLastByte)
                {
                    var length = (int)Math.Min(buffer.Length, afterLastByte - curStart);
                    reader.Read(buffer, 0, length);
                    storage.CopyToStorage(curStart, bufferHandle.AddrOfPinnedObject(), length);
                    curStart += length;
                }
            }
            finally
            {
                bufferHandle.Free();
            }
        }
    }
}
