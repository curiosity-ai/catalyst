using System;
using System.Buffers;

namespace Catalyst.Algorithms
{
    public static class Levenshtein
    {
        // OSA needs up to three rows, so keep this conservative:
        // 3 * 256 * sizeof(int) ~= 3 KB on the stack.
        private const int StackallocLimit = 256;

        private static int Minimum(int a, int b, int c)
        {
            int min = a < b ? a : b;
            return min < c ? min : c;
        }

        /// <summary>
        /// Computes the Levenshtein edit distance between two spans.
        /// When <paramref name="allowAdjacentTransposition"/> is true,
        /// adjacent transpositions are counted as one operation using the
        /// restricted Damerau-Levenshtein / Optimal String Alignment variant.
        /// </summary>
        public static int GetDistance(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target,
            bool allowAdjacentTransposition = false)
        {
            if (source.SequenceEqual(target))
                return 0;

            if (source.Length == 0)
                return target.Length;

            if (target.Length == 0)
                return source.Length;

            // Use the shorter input as the column dimension to minimize row-buffer size.
            // Levenshtein and the adjacent-transposition variant are symmetric.
            if (target.Length > source.Length)
            {
                ReadOnlySpan<char> temp = source;
                source = target;
                target = temp;
            }

            return allowAdjacentTransposition
                ? GetOptimalStringAlignmentDistance(source, target)
                : GetLevenshteinDistance(source, target);
        }

        private static int GetLevenshteinDistance(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target)
        {
            int columns = target.Length + 1;

            if (columns <= StackallocLimit)
            {
                Span<int> previous = stackalloc int[columns];
                Span<int> current = stackalloc int[columns];
                return GetLevenshteinDistanceCore(source, target, previous, current);
            }

            int[] previousArray = ArrayPool<int>.Shared.Rent(columns);
            int[] currentArray = ArrayPool<int>.Shared.Rent(columns);

            try
            {
                return GetLevenshteinDistanceCore(
                    source,
                    target,
                    previousArray.AsSpan(0, columns),
                    currentArray.AsSpan(0, columns));
            }
            finally
            {
                ArrayPool<int>.Shared.Return(previousArray);
                ArrayPool<int>.Shared.Return(currentArray);
            }
        }

        private static int GetLevenshteinDistanceCore(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target,
            Span<int> previous,
            Span<int> current)
        {
            int columns = target.Length + 1;

            for (int j = 0; j < columns; j++)
                previous[j] = j;

            for (int i = 1; i <= source.Length; i++)
            {
                current[0] = i;

                char sourceChar = source[i - 1];

                for (int j = 1; j < columns; j++)
                {
                    int substitutionCost = sourceChar == target[j - 1] ? 0 : 1;

                    current[j] = Minimum(
                        previous[j] + 1,                    // deletion
                        current[j - 1] + 1,                 // insertion
                        previous[j - 1] + substitutionCost); // substitution
                }

                Span<int> temp = previous;
                previous = current;
                current = temp;
            }

            return previous[target.Length];
        }

        private static int GetOptimalStringAlignmentDistance(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target)
        {
            int columns = target.Length + 1;

            if (columns <= StackallocLimit)
            {
                Span<int> previousPrevious = stackalloc int[columns];
                Span<int> previous = stackalloc int[columns];
                Span<int> current = stackalloc int[columns];
                return GetOptimalStringAlignmentDistanceCore(
                    source, target, previousPrevious, previous, current);
            }

            int[] previousPreviousArray = ArrayPool<int>.Shared.Rent(columns);
            int[] previousArray = ArrayPool<int>.Shared.Rent(columns);
            int[] currentArray = ArrayPool<int>.Shared.Rent(columns);

            try
            {
                return GetOptimalStringAlignmentDistanceCore(
                    source,
                    target,
                    previousPreviousArray.AsSpan(0, columns),
                    previousArray.AsSpan(0, columns),
                    currentArray.AsSpan(0, columns));
            }
            finally
            {
                ArrayPool<int>.Shared.Return(previousPreviousArray);
                ArrayPool<int>.Shared.Return(previousArray);
                ArrayPool<int>.Shared.Return(currentArray);
            }
        }

        private static int GetOptimalStringAlignmentDistanceCore(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target,
            Span<int> previousPrevious,
            Span<int> previous,
            Span<int> current)
        {
            int columns = target.Length + 1;

            for (int j = 0; j < columns; j++)
                previous[j] = j;

            for (int i = 1; i <= source.Length; i++)
            {
                current[0] = i;

                char sourceChar = source[i - 1];

                for (int j = 1; j < columns; j++)
                {
                    int substitutionCost = sourceChar == target[j - 1] ? 0 : 1;

                    int best = Minimum(
                        previous[j] + 1,                    // deletion
                        current[j - 1] + 1,                 // insertion
                        previous[j - 1] + substitutionCost); // substitution

                    if (i > 1 &&
                        j > 1 &&
                        sourceChar == target[j - 2] &&
                        source[i - 2] == target[j - 1])
                    {
                        best = Math.Min(best, previousPrevious[j - 2] + 1);
                    }

                    current[j] = best;
                }

                Span<int> temp = previousPrevious;
                previousPrevious = previous;
                previous = current;
                current = temp;
            }

            return previous[target.Length];
        }
    }
}
