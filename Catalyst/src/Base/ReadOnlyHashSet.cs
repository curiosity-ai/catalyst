using System.Collections;
using System.Collections.Generic;

namespace Catalyst
{
    public class ReadOnlyHashSet<T> : IEnumerable<T>
    {
        private readonly HashSet<T> Set;

        public ReadOnlyHashSet(HashSet<T> set)
        {
            Set = set;
        }

        public bool Contains(T item) => Set.Contains(item);

        public IEnumerator<T> GetEnumerator() => Set.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => Set.GetEnumerator();
    }
}