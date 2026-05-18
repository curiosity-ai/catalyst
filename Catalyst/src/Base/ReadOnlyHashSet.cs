using MessagePack;
using System.Collections;
using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Represents a read-only wrapper around a <see cref="HashSet{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of elements in the set.</typeparam>
    public class ReadOnlyHashSet<T> : IEnumerable<T>
    {
        private HashSet<T> Set;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadOnlyHashSet{T}"/> class.
        /// </summary>
        /// <param name="set">The set to wrap.</param>
        public ReadOnlyHashSet(HashSet<T> set)
        {
            Set = set;
        }

        /// <summary>
        /// Determines whether the set contains a specific value.
        /// </summary>
        /// <param name="item">The object to locate.</param>
        /// <returns>True if the item is found, false otherwise.</returns>
        public bool Contains(T item) => Set.Contains(item);

        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator for the set.</returns>
        public IEnumerator<T> GetEnumerator() => Set.GetEnumerator();

        /// <summary>
        /// Returns an enumerator that iterates through the collection.
        /// </summary>
        /// <returns>An enumerator for the set.</returns>
        IEnumerator IEnumerable.GetEnumerator() => Set.GetEnumerator();
    }
}