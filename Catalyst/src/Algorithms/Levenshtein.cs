using System;
using System.Buffers;

namespace Catalyst.Algorithms
{
    internal static class Levenshtein
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

            int[] previousArray = Array.Empty<int>();
            int[] currentArray = Array.Empty<int>();

            Span<int> previous = default;
            Span<int> current = default;

            try
            {
                if (columns <= StackallocLimit)
                {
                    previous = stackalloc int[columns];
                    current = stackalloc int[columns];
                }
                else
                {
                    previousArray = ArrayPool<int>.Shared.Rent(columns);
                    currentArray = ArrayPool<int>.Shared.Rent(columns);

                    previous = previousArray.AsSpan(0, columns);
                    current = currentArray.AsSpan(0, columns);
                }

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
            finally
            {
                if (previousArray.Length != 0)
                    ArrayPool<int>.Shared.Return(previousArray);

                if (currentArray.Length != 0)
                    ArrayPool<int>.Shared.Return(currentArray);
            }
        }

        private static int GetOptimalStringAlignmentDistance(
            ReadOnlySpan<char> source,
            ReadOnlySpan<char> target)
        {
            int columns = target.Length + 1;

            int[] previousPreviousArray = Array.Empty<int>();
            int[] previousArray = Array.Empty<int>();
            int[] currentArray = Array.Empty<int>();

            Span<int> previousPrevious = default;
            Span<int> previous = default;
            Span<int> current = default;

            try
            {
                if (columns <= StackallocLimit)
                {
                    previousPrevious = stackalloc int[columns];
                    previous = stackalloc int[columns];
                    current = stackalloc int[columns];
                }
                else
                {
                    previousPreviousArray = ArrayPool<int>.Shared.Rent(columns);
                    previousArray = ArrayPool<int>.Shared.Rent(columns);
                    currentArray = ArrayPool<int>.Shared.Rent(columns);

                    previousPrevious = previousPreviousArray.AsSpan(0, columns);
                    previous = previousArray.AsSpan(0, columns);
                    current = currentArray.AsSpan(0, columns);
                }

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
            finally
            {
                if (previousPreviousArray.Length != 0)
                    ArrayPool<int>.Shared.Return(previousPreviousArray);

                if (previousArray.Length != 0)
                    ArrayPool<int>.Shared.Return(previousArray);

                if (currentArray.Length != 0)
                    ArrayPool<int>.Shared.Return(currentArray);
            }
        }
    }
}