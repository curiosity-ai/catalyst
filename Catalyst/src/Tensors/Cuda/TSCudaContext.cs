using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Catalyst.Tensors.CUDA.ContextState;
using Catalyst.Tensors.CUDA.Util;

namespace Catalyst.Tensors.CUDA
{
    public struct ScratchSpace
    {
        public int size;
        public CUdeviceptr buffer;
    }

    [Serializable]
    public class TSCudaContext : IDisposable
    {
        public const int MaxDims = 25;
        private const string CacheDir = @"cuda_cache\general";


      //  private readonly int deviceCount;
        private readonly DeviceState[] devices;
        private readonly bool[,] p2pAccess;
        private readonly int[] deviceIds;

        private readonly RuntimeCompiler.KernelDiskCache diskCache;

        private readonly RuntimeCompiler.CudaCompiler compiler;
        private readonly CudaKernelCache kernelCache = new CudaKernelCache();
        

        public TSCudaContext(int[] deviceIds)
        {
            //try
            //{
            //    this.deviceCount = CudaContext.GetDeviceCount();
            //}
            //catch
            //{
            //    // CudaContext.GetDeviceCount() throws if CUDA drivers are not installed
            //    this.deviceCount = 0;
            //}

            this.deviceIds = deviceIds;

            devices = new DeviceState[deviceIds.Length];
            for (int i = 0; i < deviceIds.Length; i++)
            {
                devices[i] = new DeviceState(deviceIds[i]);
            }


       //     if (deviceCount > 0)
         //   {
                p2pAccess = EnablePeerAccess(devices.Select(x => x.CudaContext).ToArray(), devices[0].CudaContext);
            //}
            //else
            //{
            //    p2pAccess = new bool[0, 0];
            //}

            diskCache = new RuntimeCompiler.KernelDiskCache(Path.Combine(Environment.CurrentDirectory, CacheDir));
            compiler = new RuntimeCompiler.CudaCompiler(diskCache);

            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        private int GetDeviceIdIndex(int id)
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


        public RuntimeCompiler.CudaCompiler Compiler { get { return compiler; } }
        public CudaKernelCache KernelCache { get { return kernelCache; } }
      //  public int DeviceCount { get { return deviceCount; } }


        public void FreeMemoryAllDevices(bool callGC = false)
        {
            foreach (var device in devices)
            {
                device.FreeMemory(callGC);
            }
        }

        public void Dispose()
        {
            kernelCache.Dispose();

            foreach (var device in devices)
            {
                device.Dispose();
            }
        }

        public void Synchronize(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            devices[idx].CudaContext.Synchronize();
        }

        public void SynchronizeAll()
        {
            foreach (var device in devices)
            {
                device.CudaContext.Synchronize();
            }
        }

        public CudaContext CudaContextForDevice(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            return devices[idx].CudaContext;
        }

        public IDeviceAllocator AllocatorForDevice(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            return devices[idx].MemoryAllocator;
        }

        public CudaContext CudaContextForTensor(Tensor tensor)
        {
            return CudaContextForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public ScratchSpace ScratchSpaceForDevice(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            return devices[idx].ScratchSpace;
        }

        public PooledObject<CudaBlas> BlasForDevice(int deviceId)
        {
            int idx = GetDeviceIdIndex(deviceId);
            return devices[idx].BlasHandles.Get();
        }

        public PooledObject<CudaBlas> BlasForTensor(Tensor tensor)
        {
            return BlasForDevice(CudaHelpers.GetDeviceId(tensor));
        }

        public bool CanAccessPeer(int srcDevice, int peerDevice)
        {
            int srcDeviceIdx = GetDeviceIdIndex(srcDevice);
            int peerDeviceIdx = GetDeviceIdIndex(peerDevice);
            return p2pAccess[srcDeviceIdx, peerDeviceIdx];
        }

        public CudaDeviceProperties DeviceInfoForContext(CudaContext cudaContext)
        {
            int idx = GetDeviceIdIndex(cudaContext.DeviceId);
            return devices[idx].DeviceInfo;
        }

        

        // Returns a matrix of [i, j] values where [i, j] is true iff device i can access device j
        private static bool[,] EnablePeerAccess(CudaContext[] cudaContexts, CudaContext restoreCurrent)
        {
            var result = new bool[cudaContexts.Length, cudaContexts.Length];

            for (int i = 0; i < cudaContexts.Length; ++i)
            {
                for (int j = 0; j < cudaContexts.Length; ++j)
                {
                    if (i == j)
                    {
                        result[i, j] = true;
                    }
                    else
                    {
                        result[i, j] = EnablePeers(cudaContexts[i], cudaContexts[j]);
                    }
                }
            }

            restoreCurrent.SetCurrent();
            return result;
        }

        private static bool EnablePeers(CudaContext src, CudaContext target)
        {
            if (!src.DeviceCanAccessPeer(target))
                return false;

            src.SetCurrent();

            try
            {
                CudaContext.EnablePeerAccess(target);
                return true;
            }
            catch
            {
                return false;
            }
        }


        public void Precompile()
        {
            Precompile(Console.Write);
        }

        public void Precompile(Action<string> precompileProgressWriter)
        {
            var assembly = Assembly.GetExecutingAssembly();
            foreach (var applyType in assembly.TypesWithAttribute<PrecompileAttribute>(true).Where(x => !x.Item1.IsAbstract))
            {
                precompileProgressWriter("Precompiling " + applyType.Item1.Name + "\n");

                var instance = (IPrecompilable)Activator.CreateInstance(applyType.Item1);
                instance.Precompile(Compiler);
            }
        }

        public void CleanUnusedPTX()
        {
            diskCache.CleanUnused();
        }
    }
}
