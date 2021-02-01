using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    //The C# runtime doesn't yet support this (being tracked here: https://github.com/dotnet/corefx/issues/26528), so we're using the candidate code for the PR (https://gist.github.com/bbartels/87c7daae28d4905c60ae77724a401b20) till they sort it out

    internal static partial class MemoryExtensions
    {
        public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span)
             => new SpanSplitEnumerator<char>(span, ' ');

        public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span, char separator)
            => new SpanSplitEnumerator<char>(span, separator);

        public static SpanSplitSequenceEnumerator<char> Split(this ReadOnlySpan<char> span, char[] separators)
            => new SpanSplitSequenceEnumerator<char>(span, separators);
    }

    internal ref struct SpanSplitEnumerator<T> where T : IEquatable<T>
    {
        private readonly ReadOnlySpan<T> _sequence;
        private readonly T _separator;
        private int _offset;
        private int _index;

        public SpanSplitEnumerator<T> GetEnumerator() => this;

        internal SpanSplitEnumerator(ReadOnlySpan<T> span, T separator)
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

            var nextIdx = slice.IndexOf(_separator);
            _index = (nextIdx != -1 ? nextIdx : slice.Length) + 1;
            return true;
        }
    }

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
