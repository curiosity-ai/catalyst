using System;
 
namespace Catalyst.Algorithms {
 
    internal static class Levenshtein {
 
        private static int Minimum(int a, int b, int c) => (a = a < b ? a : b) < c ? a : c;
 
        /// <summary>
        /// @Wikipedia (/wiki/Levenshtein_distance, /wiki/Damerau–Levenshtein_distance
        /// Levenshtein distance is a way to calculate how different two words (or longer chains of symbols, like sentences or paragraphs) are from one another.
        /// 
        /// The simple way this works is by counting how many times you need to change one word to turn it into another word.
        /// The three atomic 'changes' considered in this measure are: inserting a single symbol (usually a character, like letter, digit etc.), deleting a single character, and replacing (substitution) a single character with another one.Moving a character to another position in the text, swapping two characters, as well as adding, deleting or replacing longer blocks of characters(like words in a sentence) are not counted as single changes in this measure.[1]
        /// For example, the Levenshtein distance between "kitten" and "sitting" is 3, since the following three edits change one into the other, and there is no way to do it with fewer than three edits:
        ///    - kitten → sitten(substitution of 's' for 'k')
        ///    - sitten → sittin(substitution of 'i' for 'e')
        ///    - sittin → sitting(insertion of 'g' at the end).
        /// 
        /// The Damerau–Levenshtein distance differs from the classical Levenshtein distance by including transpositions among its allowable operations in addition to the three classical single-character edit operations (insertions, deletions and substitutions). 
        /// 
        /// </summary>
        /// <param name="src"></param>
        /// <param name="dest"></param>
        /// <returns>Zahl (Distanz)</returns>
        public static int GetDistance(ReadOnlySpan<char> src, ReadOnlySpan<char> dest, bool damerau) {
            int n = src.Length + 1,
                m = dest.Length + 1;
 
            int[,] d = new int[src.Length + 1, dest.Length + 1];
            int i, j, cost;
 
            for (i = 0; i < n; i++) {
                d[i, 0] = i;
            }
            for (j = 0; j < m; j++) {
                d[0, j] = j;
            }
            for (i = 1; i < n; i++) {
                for (j = 1; j < m; j++) {
 
                    if (src[i - 1] == dest[j - 1])
                        cost = 0;
                    else
                        cost = 1;
 
                    d[i, j] =
                        Minimum(
                            d[i - 1, j] + 1,          // delete
                            d[i, j - 1] + 1,          // insert
                            d[i - 1, j - 1] + cost);  // replacement
 
                    if (damerau) {                    // permutation
                        if ((i > 1) && (j > 1) && (src[i - 1] == dest[j - 2]) && (src[i - 2] == dest[j - 1])) {
                            d[i, j] = Math.Min(d[i, j], d[i - 2, j - 2] + cost);
                        }
                    }
                }
            }
 
            return d[n - 1, m - 1];
        }
    }
 
}