using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;


namespace Catalyst
{
    public static class StringExtensions
    {

        public static ObjectPool<StringBuilder> StringBuilderPool = new ObjectPool<StringBuilder>(() => new StringBuilder(), 20, sb =>
        {
            if (sb.Length < 1_000_000)
            {
                sb.Length = 0; //Don't need to lose the the internal buffer for up to 1MB 
            }
            else
            {
                sb.Clear();
            }
        });


        public static string RemoveControlCharacters(this string text)
        {
            return text.AsSpan().RemoveControlCharacters();
        }

        public static string RemoveControlCharacters(this ReadOnlySpan<char> text)
        {
            if (text == null) return null;
            StringBuilder newString = new StringBuilder();
            char ch;
            for (int i = 0; i < text.Length; i++)
            {
                ch = text[i];
                if (ch == '\r' || ch == '\n' || !char.IsControl(ch)) //keep new lines
                {
                    if (ch != '\uF022')
                    {
                        newString.Append(ch);
                    }
                    else
                    {
                        newString.Append(' ');
                    }

                }
            }
            return newString.ToString();
        }

        public static string RemoveQuotesAndTrimWhiteSpace(this string text)
        {
            return text.AsSpan().RemoveQuotesAndTrimWhiteSpace();
        }

        private static readonly HashSet<char> QuotesCharacters = new HashSet<char>(new char[] { '\'', '"', '”', '“', '`', '‘', '´', '‘', '’', '‚', '„', '»', '«', '「', '」', '『', '』', '（', '）', '〔', '〕', '【', '】', '《', '》', '〈', '〉' });

        public static string RemoveQuotesAndTrimWhiteSpace(this ReadOnlySpan<char> text)
        {
            if (text == null) return null;
            StringBuilder newString = new StringBuilder();
            char ch; bool started = false;
            char last = '0';
            bool isLastWhitespace = false;
            bool isCurWhitespace = false;
            for (int i = 0; i < text.Length; i++)
            {
                ch = text[i];
                if (!QuotesCharacters.Contains(ch))
                {
                    isCurWhitespace = char.IsWhiteSpace(ch);
                    if (!started && isCurWhitespace) { continue; } // Remove any leading spaces
                    started = true;


                    if (isLastWhitespace && isCurWhitespace)
                    {
                        //Remove duplicated white-spaces in the middle;
                    }
                    else
                    {
                        newString.Append(ch);
                        last = ch;
                        isLastWhitespace = isCurWhitespace;
                    }
                }
            }
            // Remove any trailing spaces
            int k = newString.Length - 1;
            while (k >= 0 && char.IsWhiteSpace(newString[k])) { newString.Length--; k--; }
            return newString.ToString();
        }


        public static string RemoveDiacritics(this string text)
        {
            var normalizedString = text.Normalize(NormalizationForm.FormD);
            var stringBuilder = new StringBuilder();

            foreach (var c in normalizedString)
            {
                var unicodeCategory = CharUnicodeInfo.GetUnicodeCategory(c);
                if (unicodeCategory != UnicodeCategory.NonSpacingMark)
                {
                    stringBuilder.Append(c);
                }
            }

            return stringBuilder.ToString().Normalize(NormalizationForm.FormC);
        }

        public static string RemoveLigatures(this string input)
        {
            if (string.IsNullOrWhiteSpace(input)) { return input; }

            if (!input.AsSpan().HasLigatures())
            {
                return input;
            }
            else
            {
                var sb = new StringBuilder(input.Length);
                var s = input.AsSpan();

                var p = 0;
                var n = s.IndexOfAny(CharacterClasses.UnicodeLatinLigatures);
                while (n >= 0)
                {
                    sb.Append(s.Slice(p, n).ToArray());
                    var i = Array.IndexOf(CharacterClasses.UnicodeLatinLigatures, s.Slice(p + n, 1)[0]);
                    sb.Append(CharacterClasses.UnicodeLatinLigatureReplacements[i]);
                    p = p + n + 1;
                    n = s.Slice(p).IndexOfAny(CharacterClasses.UnicodeLatinLigatures);
                }
                if (p < s.Length)
                {
                    sb.Append(s.Slice(p).ToArray());
                }

                return sb.ToString();
            }
        }

        public static string ToTitleCase(this string str)
        {
            if (string.IsNullOrWhiteSpace(str)) { return str; }

            var sb = new StringBuilder(str.Length);
            char prev = str[0];
            sb.Append(char.ToUpperInvariant(prev));
            for (int i = 1; i < str.Length; i++)
            {
                char cur = str[i];
                if (char.IsWhiteSpace(prev) || (char.IsPunctuation(prev) && prev != '\''))
                {
                    cur = char.ToUpperInvariant(cur);
                }
                else
                {
                    cur = char.ToLowerInvariant(cur);
                }
                sb.Append(cur);
                prev = cur;
            }
            return sb.ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int IndexOfAny(this ReadOnlySpan<char> span, char[] anyOf, int startIndex)
        {
            if (span.Length == 0)
                return 0;  // A zero-length sequence is always treated as "found" at the start of the search space.

            int index = -1;
            int searchSpaceLength = span.Length - startIndex;
            for (int i = 0; i < anyOf.Length; i++)
            {
                var tempIndex = span.Slice(startIndex, searchSpaceLength).IndexOf(anyOf[i]);
                if ((uint)tempIndex < (uint)index)
                {
                    index = tempIndex;
                    // Reduce space for search, cause we don't care if we find the search value after the index of a previously found value
                    searchSpaceLength = tempIndex;

                    if (index == 0) break;
                }
            }
            return index < 0 ? index : index + startIndex;
        }
    }
}
