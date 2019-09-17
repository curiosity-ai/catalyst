using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.ContextState
{
    [Serializable]
    public class CudaKernelCache : IDisposable
    {
        private Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel> activeKernels = new Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel>();

        public CudaKernelCache()
        {
        }

        private object locker = new object();

        public void Dispose()
        {
            lock (locker)
            {
                foreach (var kvp in activeKernels)
                {
                    var ctx = kvp.Key.Item1;
                    var kernel = kvp.Value;

                    ctx.UnloadKernel(kernel);
                }
            }
        }

      

        public CudaKernel Get(CudaContext context, byte[] ptx, string kernelName)
        {
            lock (locker)
            {
                CudaKernel value;
                if (activeKernels.TryGetValue(Tuple.Create(context, ptx, kernelName), out value))
                {
                    return value;
                }
                else
                {
                    value = context.LoadKernelPTX(ptx, kernelName);
                    activeKernels.Add(Tuple.Create(context, ptx, kernelName), value);
                    return value;
                }
            }
        }
    }

}
