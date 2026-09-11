using System;
using System.Runtime.CompilerServices;
using System.Text;

namespace Catalyst.Models
{
    /// <summary>
    /// Turns the strings a spotter is given into the single normalized form everything else works with:
    /// trimmed, words separated by exactly one space, UTF-8, lower-cased when the model ignores case.
    ///
    /// A multi-token entry is stored as one string with its separators, which is what lets one dictionary
    /// replace the old single-token table plus the per-position multi-gram tables.
    /// </summary>
    internal static class EntryText
    {
        public const byte SEPARATOR = (byte)' ';

        /// <summary>Worst-case UTF-8 byte count for a char count, with room for a separator.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int MaxUtf8Bytes(int charCount) => charCount * 3 + 4;

        /// <summary>
        /// Normalizes an entry into <paramref name="destination"/>. Returns the byte length, or -1 when the
        /// entry holds no words or does not fit.
        /// </summary>
        public static int Normalize(ReadOnlySpan<char> entry, bool ignoreCase, Span<char> charBuffer, Span<byte> destination, out int wordCount)
        {
            wordCount = 0;
            int written = 0;
            int i       = 0;

            while (i < entry.Length)
            {
                while (i < entry.Length && entry[i] == ' ') { i++; }
                if (i >= entry.Length) { break; }

                int start = i;
                while (i < entry.Length && entry[i] != ' ') { i++; }

                if (wordCount > 0)
                {
                    if (written >= charBuffer.Length) { return -1; }
                    charBuffer[written++] = ' ';
                }
                for (int j = start; j < i; j++)
                {
                    if (written >= charBuffer.Length) { return -1; }
                    charBuffer[written++] = ignoreCase ? char.ToLowerInvariant(entry[j]) : entry[j];
                }
                wordCount++;
            }

            if (wordCount == 0) { return -1; }
            return Encode(charBuffer.Slice(0, written), destination);
        }

        /// <summary>Encodes one token - which never contains a space - into UTF-8. Returns the length, or -1 when it does not fit.</summary>
        public static int EncodeToken(ReadOnlySpan<char> token, bool ignoreCase, Span<char> charBuffer, Span<byte> destination)
        {
            if (token.Length == 0 || token.Length > charBuffer.Length) { return -1; }

            for (int i = 0; i < token.Length; i++)
            {
                charBuffer[i] = ignoreCase ? char.ToLowerInvariant(token[i]) : token[i];
            }
            return Encode(charBuffer.Slice(0, token.Length), destination);
        }

        private static int Encode(ReadOnlySpan<char> source, Span<byte> destination)
        {
            // Fast path: the catalogues this matters for are pure ASCII.
            bool ascii = true;
            for (int i = 0; i < source.Length; i++)
            {
                if (source[i] > 0x7F) { ascii = false; break; }
            }

            if (ascii)
            {
                if (source.Length > destination.Length) { return -1; }
                for (int i = 0; i < source.Length; i++) { destination[i] = (byte)source[i]; }
                return source.Length;
            }

            if (Encoding.UTF8.GetByteCount(source) > destination.Length) { return -1; }
            return Encoding.UTF8.GetBytes(source, destination);
        }

        /// <summary>64-bit hash of an already-normalized entry, matching what the prefilter is built from.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Hash(ReadOnlySpan<byte> normalized)
        {
            ulong hash = 3074457345618258791ul;
            for (int i = 0; i < normalized.Length; i++)
            {
                hash += normalized[i];
                hash *= 3074457345618258799ul;
            }
            return hash;
        }

        /// <summary>The part of a normalized entry before its first separator - what a first token has to match.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ReadOnlySpan<byte> FirstWord(ReadOnlySpan<byte> normalized)
        {
            int at = normalized.IndexOf(SEPARATOR);
            return at < 0 ? normalized : normalized.Slice(0, at);
        }
    }

    /// <summary>
    /// Blocked Bloom filter over the first word of every stored entry, consulted before the dictionary.
    ///
    /// Walking a sorted dictionary costs several cache misses where a hash table costs one, and in real text
    /// almost every token matches nothing. Every key touches bits inside a single 512-bit block, so a miss
    /// costs one cache line; the filter never reports a stored key as absent, so putting it in front of the
    /// dictionary keeps the result exact while ordinary text is rejected at hash-table speed.
    ///
    /// It is derived from the dictionary and rebuilt on load rather than serialized.
    /// </summary>
    internal sealed class EntryPrefilter
    {
        private const int BITS_PER_KEY = 10;
        private const int PROBES       = 7;

        private readonly ulong[] _bits;
        private readonly int     _blocks;

        public long EstimatedBytes => 40L + 24L + (long)_bits.Length * sizeof(ulong);
        public int  Objects        => 2;

        public static readonly EntryPrefilter Empty = new EntryPrefilter(0);

        private EntryPrefilter(int keys)
        {
            _blocks = Math.Max(1, (int)(((long)keys * BITS_PER_KEY + 511) / 512));
            _bits   = new ulong[(long)_blocks * 8];
        }

        /// <summary>Builds the filter by walking the dictionary once.</summary>
        public static EntryPrefilter Build(EntryDictionary dictionary)
        {
            if (dictionary is null || dictionary.Count == 0) { return Empty; }

            var filter = new EntryPrefilter(dictionary.Count);
            dictionary.Visit((rank, entry) => filter.Add(EntryText.Hash(EntryText.FirstWord(entry))));
            return filter;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong Mix(ulong x)
        {
            x ^= x >> 33; x *= 0xff51afd7ed558ccdUL;
            x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53UL;
            x ^= x >> 33;
            return x;
        }

        private void Add(ulong key)
        {
            ulong mixed = Mix(key);
            long  block = (long)((uint)(mixed >> 32) % (uint)_blocks) * 8;
            ulong bits  = Mix(mixed ^ 0x9E3779B97F4A7C15UL);
            for (int i = 0; i < PROBES; i++)
            {
                int bit = (int)(bits & 511);
                bits >>= 9;
                _bits[block + (bit >> 6)] |= 1UL << (bit & 63);
            }
        }

        /// <summary>False means no stored entry starts with this token; true means the dictionary has to decide.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MayStart(ulong key)
        {
            ulong mixed = Mix(key);
            long  block = (long)((uint)(mixed >> 32) % (uint)_blocks) * 8;
            ulong bits  = Mix(mixed ^ 0x9E3779B97F4A7C15UL);
            for (int i = 0; i < PROBES; i++)
            {
                int bit = (int)(bits & 511);
                bits >>= 9;
                if ((_bits[block + (bit >> 6)] & (1UL << (bit & 63))) == 0) { return false; }
            }
            return true;
        }
    }
}
