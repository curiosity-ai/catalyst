using System;

namespace Catalyst
{
    internal ref struct SpanSplitSequenceEnumerator<T> where T : IEquatable<T>
    {
        private readonly ReadOnlySpan<T> _sequence;
        private readonly ReadOnlySpan<T> _separator;
        private int _offset;
        private int _index;

        public SpanSplitSequenceEnumerator<T> GetEnumerator() => this;

        internal SpanSplitSequenceEnumerator(ReadOnlySpan<T> span, ReadOnlySpan<T> separator)
        {
            _sequence = span;
            _separator = separator;
            _index = 0;
            _offset = 0;
        }

        public Range Current => new Range(_offset, _offset + _index - 1);

        public bool MoveNext()
        {
            if (_sequence.Length - _offset < _index) { return false; }
            var slice = _sequence.Slice(_offset += _index);

            var nextIdx = slice.IndexOfAny(_separator);
            _index = (nextIdx != -1 ? nextIdx : slice.Length) + 1;
            return true;
        }
    }
}