using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Catalyst.Tensors
{
    /// <summary>
    /// Provides a thread safe reference counting implementation. Inheritors need only implement the Destroy() method,
    /// which will be called when the reference count reaches zero. The reference count automatically starts at 1.
    /// </summary>
  
    [Serializable]
    public abstract class RefCounted
    {
        private int refCount = 1;

        /// <summary>
        /// Construct a new reference counted object. The reference count automatically starts at 1.
        /// </summary>
        public RefCounted()
        {
        }

        ~RefCounted()
        {
            if (refCount > 0)
            {
                Destroy();
                refCount = 0;
            }
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected abstract void Destroy();

        /// <summary>
        /// Returns true if the object has already been destroyed; false otherwise.
        /// </summary>
        /// <returns>true if the object is destroyed; false otherwise.</returns>
        protected bool IsDestroyed()
        {
            return refCount == 0;
        }

        /// <summary>
        /// Throws an exception if the object has been destroyed, otherwise does nothing.
        /// </summary>
        protected void ThrowIfDestroyed()
        {
            if (IsDestroyed())
            {
                throw new InvalidOperationException("Reference counted object has been destroyed");
            }
        }

        protected int GetCurrentRefCount()
        {
            return refCount;
        }

        /// <summary>
        /// Increments the reference count. If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void AddRef()
        {
            int curRefCount;
            int original;
            var spin = new SpinWait();
            while (true)
            {
                curRefCount = refCount;
                if (curRefCount == 0) throw new InvalidOperationException("Cannot AddRef - object has already been destroyed");
                var desiredRefCount = curRefCount + 1;
                original = Interlocked.CompareExchange(ref refCount, desiredRefCount, curRefCount);
                if (original == curRefCount) break;
                spin.SpinOnce();
            }
        }

        /// <summary>
        /// Decrements the reference count. If the reference count reaches zero, the object is destroyed.
        /// If the object has previously been destroyed, an exception is thrown.
        /// </summary>
        public void Release()
        {
            int original;
            int curRefCount;
            var spin = new SpinWait();
            while (true)
            {
                curRefCount = refCount;
                if (curRefCount == 0) throw new InvalidOperationException("Cannot release object - object has already been destroyed");
                var desiredRefCount = refCount - 1;
                original = Interlocked.CompareExchange(ref refCount, desiredRefCount, curRefCount);
                if (original == curRefCount) break;
                spin.SpinOnce();
            }

            if (refCount <= 0)
                Destroy();
        }
    }
}
