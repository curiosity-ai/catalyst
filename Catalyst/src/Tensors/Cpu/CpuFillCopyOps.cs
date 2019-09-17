using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    [OpsClass]
    public class CpuFillCopyOps
    {
        public CpuFillCopyOps()
        {
        }


        private MethodInfo fill_func = NativeWrapper.GetMethod("TS_Fill");
        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        public void Fill(Tensor result, float value)
        {
            NativeWrapper.InvokeTypeMatch(fill_func, result, value);
        }


        private MethodInfo copy_func = NativeWrapper.GetMethod("TS_Copy");
        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        public void Copy(Tensor result, Tensor src)
        {
            if (result.ElementCount() != src.ElementCount())
                throw new InvalidOperationException("Tensors must have equal numbers of elements");
            NativeWrapper.Invoke(copy_func, result, src);
        }
    }
}
