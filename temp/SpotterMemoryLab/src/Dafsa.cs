using System;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Minimal deterministic acyclic finite-state automaton over the sorted entries, built with Daciuk's
    /// incremental algorithm. This is the "regex-style state machine" option: it recognises exactly the
    /// stored set, with no hashing and no false positives, and it shares both prefixes (like a trie) and
    /// identical suffix languages (unlike a trie) - which is what collapses a catalogue full of families
    /// that stock the same dash numbers.
    ///
    /// It also carries a per-transition count of the strings accepted before it, which turns the automaton
    /// into a minimal perfect hash: walking a key yields its lexicographic rank, so a tagged spotter can
    /// index a dense UID array by rank instead of storing keys and values in hash-table slots.
    ///
    /// Everything is in flat arrays - six of them - so the object count does not grow with the key count.
    /// </summary>
    internal sealed class Dafsa
    {
        // Registered (minimised) states, children always registered before parents so a target id is always
        // smaller than the id of the state pointing at it.
        private int[]   _start;     // [_stateCount + 1] first transition index of each state
        private byte[]  _tsym;      // [_transCount] transition symbol (1..K, mapped alphabet)
        private int[]   _ttarget;   // [_transCount]
        private ulong[] _finalBits; // [_stateCount/64]
        private int[]   _tskip;     // [_transCount] strings accepted before taking this transition
        private int     _stateCount;
        private int     _transCount;
        private int     _root;

        private readonly byte[] _map = new byte[256]; // byte -> symbol
        private int _alphabet;

        public int  States      => _stateCount;
        public int  Transitions => _transCount;
        public int  Alphabet    => _alphabet;
        public int  Root        => _root;
        public int  TrieStates  { get; private set; }
        public int  Count       { get; private set; }

        /// <summary>Plain-array form, membership only: state offsets + symbols + targets + final bitmap.</summary>
        public long BytesPlainMembership =>
            Sz.Arr(_stateCount + 1, 4) + Sz.Arr(_transCount, 1) + Sz.Arr(_transCount, 4) + Sz.Bits(_stateCount);

        /// <summary>Plain-array form plus the rank counters needed to use it as a minimal perfect hash.</summary>
        public long BytesPlainRanked => BytesPlainMembership + Sz.Arr(_transCount, 4);

        /// <summary>
        /// Classic packed DAWG encoding: one 32-bit word per transition holding symbol (6 bits), target
        /// transition index (24 bits), an end-of-list flag and a target-is-final flag. No state array and no
        /// final bitmap - only viable while the transition count fits 24 bits and the alphabet fits 6.
        /// </summary>
        public bool CanPack32   => _transCount < (1 << 24) && _alphabet <= 63;
        public long BytesPacked32 => Sz.Arr(_transCount, 4);
        public long BytesPacked32Ranked => Sz.Arr(_transCount, 4) + Sz.Arr(_transCount, 4);

        public int ObjectsPlain => 5;

        public Dafsa(StringBlob blob, int[] sortedOrder)
        {
            BuildAlphabet(blob);

            int maxLen = 0;
            for (int i = 0; i < sortedOrder.Length; i++) { maxLen = Math.Max(maxLen, blob.LengthOf(sortedOrder[i])); }

            int depth = maxLen + 2;
            var tmpSym    = new byte[depth][];
            var tmpTarget = new int[depth][];
            var tmpCount  = new int[depth];
            var tmpFinal  = new bool[depth];
            for (int d = 0; d < depth; d++) { tmpSym[d] = new byte[_alphabet + 1]; tmpTarget[d] = new int[_alphabet + 1]; }

            _start     = new int[1024];
            _tsym      = new byte[1024];
            _ttarget   = new int[1024];
            _finalBits = new ulong[16];
            _tskip     = Array.Empty<int>();
            _start[0]  = 0;

            InitRegister(Math.Max(1024, sortedOrder.Length / 4));

            var prev   = new byte[maxLen + 1];
            int prevLen = 0;
            long trieStates = 1;

            foreach (var entryIndex in sortedOrder)
            {
                var raw = blob[entryIndex];
                int cp  = 0;
                while (cp < prevLen && cp < raw.Length && _map[raw[cp]] == prev[cp]) { cp++; }

                for (int d = prevLen; d > cp; d--)
                {
                    int id = ReplaceOrRegister(tmpSym[d], tmpTarget[d], tmpCount[d], tmpFinal[d]);
                    tmpTarget[d - 1][tmpCount[d - 1] - 1] = id;
                }

                for (int d = cp; d < raw.Length; d++)
                {
                    byte sym = _map[raw[d]];
                    tmpSym[d][tmpCount[d]]    = sym;
                    tmpTarget[d][tmpCount[d]] = -1;
                    tmpCount[d]++;
                    tmpCount[d + 1] = 0;
                    tmpFinal[d + 1] = false;
                    prev[d]         = sym;
                    trieStates++;
                }
                tmpFinal[raw.Length] = true;
                prevLen              = raw.Length;
                Count++;
            }

            for (int d = prevLen; d > 0; d--)
            {
                int id = ReplaceOrRegister(tmpSym[d], tmpTarget[d], tmpCount[d], tmpFinal[d]);
                tmpTarget[d - 1][tmpCount[d - 1] - 1] = id;
            }
            _root = ReplaceOrRegister(tmpSym[0], tmpTarget[0], tmpCount[0], tmpFinal[0]);

            TrieStates = (int)Math.Min(int.MaxValue, trieStates);

            Array.Resize(ref _start, _stateCount + 1);
            Array.Resize(ref _tsym, _transCount);
            Array.Resize(ref _ttarget, _transCount);
            Array.Resize(ref _finalBits, (_stateCount + 63) / 64);
            DropRegister();
            ComputeRanks();
        }

        private void BuildAlphabet(StringBlob blob)
        {
            var used = blob.Alphabet();
            _alphabet = used.Length;
            for (int i = 0; i < used.Length; i++) { _map[used[i]] = (byte)(i + 1); }
        }

        // ---- register of minimised states ------------------------------------------------------------

        private ulong[] _regKeys;
        private int[]   _regVals;
        private int[]   _regNext; // chain of states sharing a signature hash, indexed by state id
        private int     _regMask;
        private int     _regUsed;

        private void InitRegister(int expected)
        {
            int m = 1024;
            while (m < expected * 2) { m <<= 1; }
            _regKeys = new ulong[m];
            _regVals = new int[m];
            _regNext = new int[1024];
            _regMask = m - 1;
        }

        private void DropRegister() { _regKeys = null; _regVals = null; _regNext = null; }

        private static ulong Mix(ulong x)
        {
            x ^= x >> 33; x *= 0xff51afd7ed558ccdUL;
            x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53UL;
            x ^= x >> 33;
            return x;
        }

        private int ReplaceOrRegister(byte[] syms, int[] targets, int count, bool final)
        {
            ulong sig = Mix(final ? 0x9E3779B97F4A7C15UL : 0x2545F4914F6CDD1DUL);
            for (int i = 0; i < count; i++) { sig = Mix(sig ^ ((ulong)syms[i] << 40) ^ (uint)targets[i]); }
            if (sig == 0) { sig = 1; }

            int slot = (int)(sig & (ulong)_regMask);
            while (_regKeys[slot] != 0)
            {
                if (_regKeys[slot] == sig)
                {
                    for (int s = _regVals[slot]; s >= 0; s = _regNext[s])
                    {
                        if (Equal(s, syms, targets, count, final)) { return s; }
                    }
                    break;
                }
                slot = (slot + 1) & _regMask;
            }

            int id = AddState(syms, targets, count, final);

            if (_regKeys[slot] == sig)
            {
                _regNext[id]  = _regVals[slot];
                _regVals[slot] = id;
            }
            else
            {
                _regNext[id]   = -1;
                _regKeys[slot] = sig;
                _regVals[slot] = id;
                if (++_regUsed * 2 > _regMask) { GrowRegister(); }
            }
            return id;
        }

        private void GrowRegister()
        {
            var oldKeys = _regKeys;
            var oldVals = _regVals;
            int m       = (_regMask + 1) * 2;
            _regKeys    = new ulong[m];
            _regVals    = new int[m];
            _regMask    = m - 1;
            for (int i = 0; i < oldKeys.Length; i++)
            {
                if (oldKeys[i] == 0) { continue; }
                int slot = (int)(oldKeys[i] & (ulong)_regMask);
                while (_regKeys[slot] != 0) { slot = (slot + 1) & _regMask; }
                _regKeys[slot] = oldKeys[i];
                _regVals[slot] = oldVals[i];
            }
        }

        private bool Equal(int state, byte[] syms, int[] targets, int count, bool final)
        {
            if (IsFinal(state) != final) { return false; }
            int lo = _start[state], hi = _start[state + 1];
            if (hi - lo != count) { return false; }
            for (int i = 0; i < count; i++)
            {
                if (_tsym[lo + i] != syms[i] || _ttarget[lo + i] != targets[i]) { return false; }
            }
            return true;
        }

        private int AddState(byte[] syms, int[] targets, int count, bool final)
        {
            int id = _stateCount++;
            if (_stateCount + 1 >= _start.Length)   { Array.Resize(ref _start, _start.Length * 2); }
            if (id / 64 >= _finalBits.Length)       { Array.Resize(ref _finalBits, _finalBits.Length * 2); }
            if (id >= _regNext.Length)              { Array.Resize(ref _regNext, _regNext.Length * 2); }
            if (_transCount + count >= _tsym.Length)
            {
                int n = Math.Max(_tsym.Length * 2, _transCount + count + 1);
                Array.Resize(ref _tsym, n);
                Array.Resize(ref _ttarget, n);
            }

            _start[id] = _transCount;
            for (int i = 0; i < count; i++)
            {
                _tsym[_transCount]      = syms[i];
                _ttarget[_transCount++] = targets[i];
            }
            _start[_stateCount] = _transCount;
            if (final) { _finalBits[id >> 6] |= 1UL << (id & 63); }
            return id;
        }

        private bool IsFinal(int state) => ((_finalBits[state >> 6] >> (state & 63)) & 1UL) != 0;

        // ---- rank counters --------------------------------------------------------------------------

        private void ComputeRanks()
        {
            _tskip      = new int[_transCount];
            var words   = new int[_stateCount];
            for (int s = 0; s < _stateCount; s++)
            {
                int acc = IsFinal(s) ? 1 : 0;
                int lo = _start[s], hi = _start[s + 1];
                for (int t = lo; t < hi; t++)
                {
                    _tskip[t] = acc;
                    acc      += words[_ttarget[t]]; // target id < s, already computed
                }
                words[s] = acc;
            }
        }

        /// <summary>Lexicographic rank of <paramref name="key"/>, or -1 when the automaton rejects it.</summary>
        public int Rank(ReadOnlySpan<byte> key)
        {
            int s = _root, ord = 0;
            for (int i = 0; i < key.Length; i++)
            {
                byte sym = _map[key[i]];
                if (sym == 0) { return -1; }
                int lo = _start[s], hi = _start[s + 1], found = -1;
                for (int t = lo; t < hi; t++)
                {
                    if (_tsym[t] == sym) { found = t; break; }
                }
                if (found < 0) { return -1; }
                ord += _tskip[found];
                s    = _ttarget[found];
            }
            return IsFinal(s) ? ord : -1;
        }

        /// <summary>True when some stored entry starts with <paramref name="prefix"/> (what the tokenizer needs).</summary>
        public bool HasPrefix(ReadOnlySpan<byte> prefix)
        {
            int s = _root;
            for (int i = 0; i < prefix.Length; i++)
            {
                byte sym = _map[prefix[i]];
                if (sym == 0) { return false; }
                int lo = _start[s], hi = _start[s + 1], found = -1;
                for (int t = lo; t < hi; t++)
                {
                    if (_tsym[t] == sym) { found = t; break; }
                }
                if (found < 0) { return false; }
                s = _ttarget[found];
            }
            return true;
        }
    }
}
