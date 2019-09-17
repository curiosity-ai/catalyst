using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple =false, Inherited =false)]
    public class PrecompileAttribute : Attribute
    {
        public PrecompileAttribute()
        {
        }
    }

    public interface IPrecompilable
    {
        void Precompile(CudaCompiler compiler);
    }

    public static class PrecompileHelper
    {
        public static void PrecompileAllFields(object instance, CudaCompiler compiler)
        {
            var type = instance.GetType();

            foreach (var field in type.GetFields())
            {
                if (typeof(IPrecompilable).IsAssignableFrom(field.FieldType))
                {
                    var precompilableField = (IPrecompilable)field.GetValue(instance);
                    Console.WriteLine("Compiling field " + field.Name);
                    precompilableField.Precompile(compiler);
                }
            }
        }
    }
}
