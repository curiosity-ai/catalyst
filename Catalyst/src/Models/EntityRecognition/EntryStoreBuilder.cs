using System;
using System.Collections.Generic;

namespace Catalyst.Models
{
    /// <summary>
    /// Accumulates the normalized entries and tokenizer-exception hashes a spotter is given while it is being
    /// trained, and turns them into the read-only structures the model keeps.
    ///
    /// Entries go into one growing byte blob with an offset table rather than one string object each, so
    /// building a ten-million-entry model does not put ten million objects on the heap first and then
    /// discard them - the build peak is close to the finished size instead of several times it.
    /// </summary>
    internal sealed class EntryStoreBuilder
    {
        private byte[] _blob;
        private int    _blobLength;
        private int[]  _offsets;
        private int    _count;
        private uint[] _exceptions;
        private int    _exceptionCount;

        public int Count => _count;

        public EntryStoreBuilder(int expectedEntries = 1024)
        {
            int n       = Math.Max(16, expectedEntries);
            _blob       = new byte[n * 16];
            _offsets    = new int[n + 1];
            _exceptions = new uint[16];
            _offsets[0] = 0;
        }

        public ReadOnlySpan<byte> EntryAt(int index) => _blob.AsSpan(_offsets[index], _offsets[index + 1] - _offsets[index]);

        /// <summary>Appends one already-normalized entry and returns its insertion index.</summary>
        public int Add(ReadOnlySpan<byte> normalized)
        {
            if (_blobLength + normalized.Length > _blob.Length)
            {
                Array.Resize(ref _blob, Math.Max(_blob.Length * 2, _blobLength + normalized.Length));
            }
            if (_count + 2 > _offsets.Length)
            {
                Array.Resize(ref _offsets, _offsets.Length * 2);
            }

            normalized.CopyTo(_blob.AsSpan(_blobLength));
            _blobLength       += normalized.Length;
            _offsets[++_count] = _blobLength;
            return _count - 1;
        }

        public void AddException(int hash)
        {
            if (_exceptionCount == _exceptions.Length) { Array.Resize(ref _exceptions, _exceptions.Length * 2); }
            _exceptions[_exceptionCount++] = unchecked((uint)hash);
        }

        public void AddExceptions(IEnumerable<int> hashes)
        {
            if (hashes is null) { return; }
            foreach (var h in hashes) { AddException(h); }
        }

        public CompactHash32Set BuildExceptions() => CompactHash32Set.Build(_exceptions, _exceptionCount);

        /// <summary>
        /// Entry indices in ascending byte order with duplicates removed. When the same entry was added more
        /// than once the *last* one survives, which is what the dictionary-assignment semantics of the old
        /// <c>Hashes[hash] = uid</c> did.
        /// </summary>
        public int[] SortDistinct(out int distinct)
        {
            var order = new int[_count];
            if (_count == 0) { distinct = 0; return order; }

            var keys = new ulong[_count];
            for (int i = 0; i < _count; i++)
            {
                order[i] = i;
                keys[i]  = PackPrefix(EntryAt(i));
            }

            Array.Sort(keys, order);

            // Entries sharing their first eight bytes land in one run; re-sort each run on the whole entry,
            // breaking exact ties by insertion order so the last one added is the one kept below.
            int start = 0;
            while (start < _count)
            {
                int end = start + 1;
                while (end < _count && keys[end] == keys[start]) { end++; }
                if (end - start > 1)
                {
                    var run = new int[end - start];
                    Array.Copy(order, start, run, 0, run.Length);
                    Array.Sort(run, CompareEntries);
                    Array.Copy(run, 0, order, start, run.Length);
                }
                start = end;
            }

            int written = 0;
            for (int i = 0; i < _count; i++)
            {
                bool lastOfRun = i + 1 >= _count || !EntryAt(order[i]).SequenceEqual(EntryAt(order[i + 1]));
                if (lastOfRun) { order[written++] = order[i]; }
            }

            distinct = written;
            return order;
        }

        public EntryDictionary BuildDictionary(int[] order, int distinct) => EntryDictionary.Build(_blob, _offsets, order, distinct);

        private int CompareEntries(int a, int b)
        {
            int cmp = EntryAt(a).SequenceCompareTo(EntryAt(b));
            return cmp != 0 ? cmp : a.CompareTo(b);
        }

        // First eight bytes, big-endian, zero-padded: a shorter entry sorts before a longer one sharing its head.
        private ulong PackPrefix(ReadOnlySpan<byte> entry)
        {
            ulong key = 0;
            for (int i = 0; i < 8; i++)
            {
                key = (key << 8) | (i < entry.Length ? entry[i] : 0u);
            }
            return key;
        }
    }
}
