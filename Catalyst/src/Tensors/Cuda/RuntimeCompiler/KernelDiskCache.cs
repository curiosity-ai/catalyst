using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace Catalyst.Tensors.CUDA.RuntimeCompiler
{

    [Serializable]
    public class KernelDiskCache
    {
        private readonly string cacheDir;
        private readonly Dictionary<string, byte[]> memoryCachedKernels = new Dictionary<string, byte[]>();


        public KernelDiskCache(string cacheDir)
        {
            this.cacheDir = cacheDir;
            if (!System.IO.Directory.Exists(cacheDir))
            {
                System.IO.Directory.CreateDirectory(cacheDir);
            }
        }

        /// <summary>
        /// Deletes all kernels from disk if they are not currently loaded into memory. Calling this after
        /// calling TSCudaContext.Precompile() will delete any cached .ptx files that are no longer needed
        /// </summary>
        public void CleanUnused()
        {
            foreach (var file in Directory.GetFiles(cacheDir))
            {
                var key = KeyFromFilePath(file);
                if (!memoryCachedKernels.ContainsKey(key))
                {
                    File.Delete(file);
                }
            }
        }
        
        public byte[] Get(string fullSourceCode, Func<string, byte[]> compile)
        {
            var key = KeyFromSource(fullSourceCode);
            byte[] ptx;
            if (memoryCachedKernels.TryGetValue(key, out ptx))
            {
                return ptx;
            }
            else if (TryGetFromFile(key, out ptx))
            {
                memoryCachedKernels.Add(key, ptx);
                return ptx;
            }
            else
            {
                ptx = compile(fullSourceCode);
                memoryCachedKernels.Add(key, ptx);
                WriteToFile(key, ptx);
                return ptx;
            }
        }


        private void WriteToFile(string key, byte[] ptx)
        {
            var filePath = FilePathFromKey(key);
            System.IO.File.WriteAllBytes(filePath, ptx);
        }

        private bool TryGetFromFile(string key, out byte[] ptx)
        {
            var filePath = FilePathFromKey(key);
            if (!System.IO.File.Exists(filePath))
            {
                ptx = null;
                return false;
            }

            ptx = System.IO.File.ReadAllBytes(filePath);
            return true;
        }

        private string FilePathFromKey(string key)
        {
            return System.IO.Path.Combine(cacheDir, key + ".ptx");
        }

        private string KeyFromFilePath(string filepath)
        {
            return Path.GetFileNameWithoutExtension(filepath);
        }

        private static string KeyFromSource(string fullSource)
        {
            var fullKey = fullSource.Length.ToString() + fullSource;

            using (var sha1 = new SHA1Managed())
            {
                return BitConverter.ToString(sha1.ComputeHash(Encoding.UTF8.GetBytes(fullKey)))
                    .Replace("-", "");
            }
        }
    }
}
