using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.Util
{
    public class PooledObject<T> : IDisposable
    {
        private readonly Action<PooledObject<T>> onDispose;
        private readonly T value;

        private bool disposed = false;

        public PooledObject(T value, Action<PooledObject<T>> onDispose)
        {
            if (onDispose == null) throw new ArgumentNullException("onDispose");

            this.onDispose = onDispose;
            this.value = value;
        }

        public T Value
        {
            get
            {
                if (disposed) throw new ObjectDisposedException(this.ToString());
                return value;
            }
        }

        public void Dispose()
        {
            if (!disposed)
            {
                onDispose(this);
                disposed = true;
            }
            else
            {
                throw new ObjectDisposedException(this.ToString());
            }
        }
    }

    public class ObjectPool<T> : IDisposable
    {
        private readonly Func<T> constructor;
        private readonly Action<T> destructor;
        private readonly Stack<T> freeList = new Stack<T>();
        private bool disposed = false;


        public ObjectPool(int initialSize, Func<T> constructor, Action<T> destructor)
        {
            if (constructor == null) throw new ArgumentNullException("constructor");
            if (destructor == null) throw new ArgumentNullException("destructor");

            this.constructor = constructor;
            this.destructor = destructor;

            for(int i = 0; i < initialSize; ++i)
            {
                freeList.Push(constructor());
            }
        }

        public void Dispose()
        {
            if (!disposed)
            {
                disposed = true;
                foreach (var item in freeList)
                {
                    destructor(item);
                }
                freeList.Clear();
            }
        }

        public PooledObject<T> Get()
        {
            T value = freeList.Count > 0 ? freeList.Pop() : constructor();
            return new PooledObject<T>(value, Release);
        }

        private void Release(PooledObject<T> handle)
        {
            freeList.Push(handle.Value);
        }
    }
}
