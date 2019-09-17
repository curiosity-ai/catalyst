using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors.Cpu
{
    public static class NativeWrapper
    {
        public static MethodInfo GetMethod(string name)
        {
            return typeof(CpuOpsNative).GetMethod(name, BindingFlags.Public | BindingFlags.Static);
        }

        public static Tensor InvokeNullableResultElementwise(MethodInfo method, params object[] args)
        {
            Tensor resultTensor;
            if(args[0] == null)
            {
                var otherTensor = args.OfType<Tensor>().First();
                resultTensor = TensorResultBuilder.GetWriteTarget(null, otherTensor, false, otherTensor.Sizes);
            }
            else
            {
                var resultSrc = (Tensor)args[0];
                var otherTensor = args.OfType<Tensor>().Skip(1).First();
                resultTensor = TensorResultBuilder.GetWriteTarget(resultSrc, otherTensor, false, otherTensor.Sizes);
            }

            args[0] = resultTensor;
            InvokeTypeMatch(method, args);
            return resultTensor;
        }

        public static Tensor InvokeNullableResultDimensionwise(MethodInfo method, Tensor result, Tensor src, int dimension, params object[] extraArgs)
        {
            if (dimension < 0 || dimension >= src.Sizes.Length) throw new ArgumentOutOfRangeException("dimension");

            var desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;
            var resultTensor = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);

            var finalArgs = new List<object>(extraArgs.Length + 3);
            finalArgs.Add(resultTensor);
            finalArgs.Add(src);
            finalArgs.Add(dimension);
            finalArgs.AddRange(extraArgs);
            InvokeTypeMatch(method, finalArgs.ToArray());
            return resultTensor;
        }

        public static void InvokeTypeMatch(MethodInfo method, params object[] args)
        {
            var tensors = args.OfType<Tensor>();
            if (tensors.Any())
            {
                var elemType = tensors.First().ElementType;
                if (!tensors.All(x => x.ElementType == elemType))
                {
                    var allTypes = string.Join(", ", tensors.Select(x => x.ElementType));
                    throw new InvalidOperationException("All tensors must have the same argument types. Given: " + allTypes);
                }
            }

            Invoke(method, args);
        }


        public static IDisposable BuildTensorRefPtr(Tensor tensor, out IntPtr tensorRefPtr)
        {
            var tensorRef = NativeWrapper.AllocTensorRef(tensor);
            var tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(tensorRef, tensorPtr, false);

            tensorRefPtr = tensorPtr;

            return new DelegateDisposable(() =>
            {
                Marshal.FreeHGlobal(tensorPtr);
                NativeWrapper.FreeTensorRef(tensorRef);
            });
        }

        public static void Invoke(MethodInfo method, params object[] args)
        {
            var freeListTensor = new List<TensorRef64>();
            var freeListPtr = new List<IntPtr>();

            try
            {
                for (int i = 0; i < args.Length; ++i)
                {
                    if (args[i] is Tensor)
                    {
                        var tensor = (Tensor)args[i];
                        if (!(tensor.Storage is CpuStorage))
                        {
                            throw new InvalidOperationException("Argument " + i + " is not a Cpu tensor");
                        }

                        var tensorRef = AllocTensorRef(tensor);
                        var tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
                        Marshal.StructureToPtr(tensorRef, tensorPtr, false);

                        args[i] = tensorPtr;

                        freeListTensor.Add(tensorRef);
                        freeListPtr.Add(tensorPtr);
                    }
                }

                //return method.Invoke(null, args);
                var result = (int)method.Invoke(null, args);
                if(result != 0)
                {
                    throw new ApplicationException(GetLastError());
                }
            }
            finally
            {
                foreach (var tensorRef in freeListTensor)
                {
                    FreeTensorRef(tensorRef);
                }

                foreach (var tensorPtr in freeListPtr)
                {
                    Marshal.FreeHGlobal(tensorPtr);
                }
            }
        }

        public static void CheckResult(int result)
        {
            if (result != 0)
            {
                throw new ApplicationException(GetLastError());
            }
        }

        private static string GetLastError()
        {
            var strPtr = CpuOpsNative.TS_GetLastError();
            return Marshal.PtrToStringAnsi(strPtr);
        }


        public static TensorRef64 AllocTensorRef(Tensor tensor)
        {
            var tensorRef = new TensorRef64();
            tensorRef.buffer = CpuNativeHelpers.GetBufferStart(tensor);
            tensorRef.dimCount = tensor.Sizes.Length;
            tensorRef.sizes = AllocArray(tensor.Sizes);
            tensorRef.strides = AllocArray(tensor.Strides);
            tensorRef.elementType = (CpuDType)tensor.ElementType;
            return tensorRef;
        }

        private static IntPtr AllocArray(long[] data)
        {
            var result = Marshal.AllocHGlobal(sizeof(long) * data.Length);
            Marshal.Copy(data, 0, result, data.Length);
            return result;
        }

        public static void FreeTensorRef(TensorRef64 tensorRef)
        {
            Marshal.FreeHGlobal(tensorRef.sizes);
            Marshal.FreeHGlobal(tensorRef.strides);
        }
    }
}
