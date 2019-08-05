// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using MessagePack;
using Mosaik.Core;
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Catalyst
{
    public class BufferedMatrix : IMatrix, IDisposable
    {
        public int Rows { get; private set; }
        public int Columns { get; private set; }

        private readonly Stream Source;
        private readonly object _lockSource = new object();
        private readonly QuantizationType Quantization;
        private readonly long BeginOfData;
        private readonly long RowSize;
        private readonly Thread CacheThread;

        private long CacheCount = 0;
        private ReaderWriterLockSlim CacheLock = new ReaderWriterLockSlim();
        private ConcurrentDictionary<int, float[]> Cache = new ConcurrentDictionary<int, float[]>();
        private CancellationTokenSource DisposingToken;

        public BufferedMatrix(Stream source, QuantizationType quantization, int cacheSize) : base()
        {
            Source = source;
            Rows = MessagePackBinary.ReadInt32(source);
            Columns = MessagePackBinary.ReadInt32(source);
            BeginOfData = source.Position;

            var tempArray = MessagePackBinary.ReadBytes(source);
            RowSize = source.Position - BeginOfData;

            Quantization = quantization;
            DisposingToken = new CancellationTokenSource();

            CacheThread = new Thread(() =>
            {
                while (!DisposingToken.Token.IsCancellationRequested)
                {
                    if (Interlocked.Read(ref CacheCount) > cacheSize)
                    {
                        CacheLock.EnterWriteLock();
                        Cache.Clear();
                        CacheCount = 0;
                        CacheLock.ExitWriteLock();
                    }
                    Thread.Sleep(1000);
                }
            });
            CacheThread.IsBackground = true; //Won't block when terminating the process
            CacheThread.Start();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ref float[] GetRowRef(int index)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetRowCopy(int index)
        {
            float[] data;

            CacheLock.EnterReadLock();

            try
            {
                if (!Cache.TryGetValue(index, out data))
                {
                    data = new float[Columns];
                    lock (_lockSource)
                    {
                        Source.Position = index * RowSize + BeginOfData;
                        switch (Quantization)
                        {
                            case QuantizationType.None:
                            {
                                var byteArray = MessagePackBinary.ReadBytes(Source);
                                System.Buffer.BlockCopy(byteArray, 0, data, 0, byteArray.Length);
                                break;
                            }
                            case QuantizationType.OneBit:
                            {
                                var byteArray = MessagePackBinary.ReadBytes(Source);
                                var bits = new BitArray(byteArray);
                                for (int j = 0; j < Columns; j++)
                                {
                                    data[j] = bits[j] ? 0.33f : -0.33f;
                                }
                                break;
                            }
                            default: throw new NotImplementedException();
                        }
                    }
                    Interlocked.Increment(ref CacheCount);
                    Cache.TryAdd(index, data);
                }
            }
            catch (Exception E)
            {
                throw E;
            }
            finally
            {
                CacheLock.ExitReadLock();
            }

            return data;
        }

        public float this[int i, int j]
        {
            get { return GetRowRef(i)[j]; }
            set { throw new NotSupportedException(); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Matrix Zero()
        {
            throw new NotSupportedException();
        }

        public Matrix Uniform(float a)
        {
            throw new NotSupportedException();
        }

        public void ResizeAndFillRows(int newRows, float a)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(float[] vec, int i, float a)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void AddToRow(ref float[] vec, int i)
        {
            throw new NotSupportedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ref float[] vec, int i)
        {
            var row = GetRowCopy(i);
            var d = SIMD.DotProduct(ref row, ref vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float DotRow(ref float[] vec, ref float[] data)
        {
            var d = SIMD.DotProduct(ref data, ref vec);
            Debug.Assert(!float.IsNaN(d));
            return d;
        }

        public Matrix Multiply(Matrix other)
        {
            throw new NotSupportedException();
        }

        public Matrix Transpose()
        {
            throw new NotSupportedException();
        }

        public void Dispose()
        {
            DisposingToken.Cancel();

            while (CacheThread.IsAlive) { Thread.Sleep(1); }
        }

        public void ToStream(Stream stream, QuantizationType quantization)
        {
            throw new NotImplementedException();
        }
    }
}