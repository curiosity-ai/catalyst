using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.Core
{
    public class DelegateDisposable : IDisposable
    {
        private readonly Action action;

        public DelegateDisposable(Action action)
        {
            this.action = action;
        }

        public virtual void Dispose()
        {
            action();
        }
    }
}
