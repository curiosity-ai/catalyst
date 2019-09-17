using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.RuntimeCompiler
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    public class CudaIncludeAttribute : Attribute
    {
        public string FieldName { get; private set; }
        public string IncludeName { get; private set; }

        public CudaIncludeAttribute(string fieldName, string includeName)
        {
            this.FieldName = fieldName;
            this.IncludeName = includeName;
        }
    }
}
