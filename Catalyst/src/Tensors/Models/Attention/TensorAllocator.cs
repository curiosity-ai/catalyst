using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Tensors;
using Catalyst.Tensors.CUDA;
using Mosaik.Core;

namespace Catalyst.Tensors.Models
{
    public class TensorAllocator
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<TensorAllocator>();

        private static IAllocator[] allocator = null;
        private static TSCudaContext cudaContext = null;
        private static int[] deviceIds;


        public static void InitDevices(int[] ids)
        {
            deviceIds = ids;

            foreach (var id in deviceIds)
            {
                Logger.LogInformation($"Initialize device '{id}'");
            }

            cudaContext = new TSCudaContext(deviceIds);
            cudaContext.Precompile(Console.Write);
            cudaContext.CleanUnusedPTX();

            allocator = new IAllocator[deviceIds.Length];
        }

        public static IAllocator Allocator(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            if (allocator[idx] == null)
            {
                allocator[idx] = new CudaAllocator(cudaContext, deviceId);
            }

            return allocator[idx];

        }

        private static int GetDeviceIdIndex(int id)
        {
            for (int i = 0; i < deviceIds.Length; i++)
            {
                if (deviceIds[i] == id)
                {
                    return i;
                }
            }

            return -1;
        }

        public static void FreeMemoryAllDevices(bool callGC = false)
        {
            GC.Collect();
            if (cudaContext != null)
            {
                cudaContext.FreeMemoryAllDevices(callGC);
            }
        }
    }
}
