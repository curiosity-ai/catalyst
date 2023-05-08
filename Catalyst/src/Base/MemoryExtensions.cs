using System;

namespace Catalyst
{
    internal static partial class MemoryExtensions
    {
        //The C# runtime doesn't yet support this (being tracked here: https://github.com/dotnet/corefx/issues/26528), so we're using the candidate code for the PR (https://gist.github.com/bbartels/87c7daae28d4905c60ae77724a401b20) till they sort it out

        public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span)
             => new SpanSplitEnumerator<char>(span, ' ');

        public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span, char separator)
            => new SpanSplitEnumerator<char>(span, separator);

        public static SpanSplitSequenceEnumerator<char> Split(this ReadOnlySpan<char> span, char[] separators)
            => new SpanSplitSequenceEnumerator<char>(span, separators);
    }
}