using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst
{
    public static class ResourceLoader
    {
        public static Stream OpenResource(Assembly assembly, string resourceFile)
        {
            return assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Resources.{resourceFile}");
        }

        public static async Task<T> LoadAsync<T>(Assembly assembly, string resourceFile, Func<Stream, Task<T>> loader)
        {
            using(var stream = OpenResource(assembly, resourceFile)) 
            {
                return await loader(stream);
            }
        }
    }
}
