using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst
{
    public static class StringSplitHelper
    {
        /// <summary>
        /// Splits an original string into perfectly matching sequential parts based on a list of approximate parts.
        /// This is useful when an LLM is asked to split a string into parts, but the returned parts have missing content,
        /// typos, or whitespace changes. The returned parts will be exact substrings of the original string, perfectly
        /// covering it from start to finish without any missing content or overlaps.
        /// </summary>
        /// <param name="originalText">The original string to split.</param>
        /// <param name="splitParts">The approximate parts (e.g. from an LLM).</param>
        /// <param name="maxMismatchPercentage">The maximum allowed mismatch percentage (0.0 to 1.0) between the original text and the concatenated parts.</param>
        /// <param name="output">A sequence of exact substrings of the original text, corresponding to each of the split parts.</param>
        /// <returns>True if the mismatch percentage is within the limit, false otherwise.</returns>
        public static bool TryRealignSplit(string originalText, IEnumerable<string> splitParts, float maxMismatchPercentage, out string[] output)
        {
            var parts = splitParts.ToList();
            if (parts.Count == 0)
            {
                output = new[] { originalText };
                return true;
            }
            if (string.IsNullOrEmpty(originalText))
            {
                output = new string[0];
                return true;
            }

            // Strip whitespace from the original text to compute alignment,
            // while retaining mapping back to the original text indices.
            var cleanToOriginal = new List<int>();
            var cleanOriginalStr = new StringBuilder();

            for (int i = 0; i < originalText.Length; i++)
            {
                if (!char.IsWhiteSpace(originalText[i]))
                {
                    cleanToOriginal.Add(i);
                    cleanOriginalStr.Append(char.ToLowerInvariant(originalText[i]));
                }
            }
            string cleanOriginal = cleanOriginalStr.ToString();

            // Strip whitespace from the parts and concatenate them,
            // tracking the index at which each part begins in the concatenated string.
            var cleanParts = new List<string>();
            int currentCleanIndex = 0;
            var partBoundariesInClean = new List<int>();

            for (int p = 0; p < parts.Count; p++)
            {
                partBoundariesInClean.Add(currentCleanIndex);
                var sb = new StringBuilder();
                foreach (char c in parts[p])
                {
                    if (!char.IsWhiteSpace(c))
                    {
                        sb.Append(char.ToLowerInvariant(c));
                        currentCleanIndex++;
                    }
                }
                cleanParts.Add(sb.ToString());
            }
            partBoundariesInClean.Add(currentCleanIndex); // End boundary
            string cleanAllParts = string.Join("", cleanParts);

            int n = cleanOriginal.Length;
            int m = cleanAllParts.Length;

            // Compute edit distance / alignment cost using Needleman-Wunsch dp table
            var dp = new int[n + 1, m + 1];
            for (int i = 0; i <= n; i++) dp[i, 0] = i;
            for (int j = 0; j <= m; j++) dp[0, j] = j;

            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    int cost = cleanOriginal[i - 1] == cleanAllParts[j - 1] ? 0 : 1;
                    dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
                }
            }

            // Since the Needleman-Wunsch DP costs for insertions, deletions, and substitutions are non-negative
            // (0 for a match, 1 otherwise), the cumulative cost dp[i, j] is monotonically increasing along any
            // valid path from (0,0) to (n,m). Therefore, dp[n, m] represents not only the total alignment
            // cost but is also mathematically guaranteed to be the maximum value found anywhere along the optimal path.
            float mismatchPercentage = dp[n, m] / (float)Math.Max(Math.Max(n, m), 1);
            if (mismatchPercentage > maxMismatchPercentage)
            {
                output = null;
                return false;
            }

            int currI = n;
            int currJ = m;
            var partStartsInOriginal = new int[parts.Count + 1];
            for (int i = 0; i < partStartsInOriginal.Length; i++) partStartsInOriginal[i] = -1;

            partStartsInOriginal[0] = 0;
            partStartsInOriginal[parts.Count] = originalText.Length;

            // Backtrack to find the optimal alignment
            while (currI > 0 || currJ > 0)
            {
                for (int p = 1; p < parts.Count; p++)
                {
                    if (currJ == partBoundariesInClean[p] && partStartsInOriginal[p] == -1)
                    {
                        if (currI < n) {
                            partStartsInOriginal[p] = cleanToOriginal[currI];
                        } else {
                            partStartsInOriginal[p] = originalText.Length;
                        }
                    }
                }

                if (currI > 0 && currJ > 0 && dp[currI, currJ] == dp[currI - 1, currJ - 1] + (cleanOriginal[currI - 1] == cleanAllParts[currJ - 1] ? 0 : 1))
                {
                    currI--;
                    currJ--;
                }
                else if (currI > 0 && dp[currI, currJ] == dp[currI - 1, currJ] + 1)
                {
                    currI--;
                }
                else
                {
                    currJ--;
                }
            }

            // Fill unassigned boundaries with the previous part's start index (left-to-right propagation)
            for (int p = 1; p < parts.Count; p++)
            {
                if (partStartsInOriginal[p] == -1)
                {
                    partStartsInOriginal[p] = partStartsInOriginal[p - 1];
                }
            }

            // Ensure boundaries are monotonically increasing and within string bounds
            for (int p = 1; p <= parts.Count; p++)
            {
                if (partStartsInOriginal[p] < partStartsInOriginal[p - 1])
                    partStartsInOriginal[p] = partStartsInOriginal[p - 1];
                if (partStartsInOriginal[p] > originalText.Length)
                    partStartsInOriginal[p] = originalText.Length;
            }

            var result = new List<string>();
            for (int p = 0; p < parts.Count; p++)
            {
                int start = partStartsInOriginal[p];
                int end = partStartsInOriginal[p + 1];

                result.Add(originalText.Substring(start, end - start));
            }

            output = result.ToArray();
            return true;
        }
    }
}
