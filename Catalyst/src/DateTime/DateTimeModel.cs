using System;
using Mosaik.Core;
using System.Buffers;
using System.Collections.Generic;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>
    /// Recognises dates, times, periods, durations and recurrences in free text and resolves them against a
    /// reference moment. The scan is regular-expression free and allocation free until something matches:
    /// the lexeme and node buffers are rented, and only a hit allocates its resolution.
    /// </summary>
    public sealed class DateTimeModel
    {
        private const int DefaultNodeCapacity = 256;

        private readonly Lexicon _lexicon;

        public DateTimeModel(Lexicon lexicon)
        {
            _lexicon = lexicon ?? throw new ArgumentNullException(nameof(lexicon));
        }

        public Lexicon Lexicon => _lexicon;

        public static DateTimeModel For(Language language, bool useUsEnglishForEnglish = false)
        {
            return new DateTimeModel(Lexicons.For(language, useUsEnglishForEnglish));
        }

        public List<DateTimeEntity> Parse(string text) => Parse(text, DateTime.Now);

        public List<DateTimeEntity> Parse(string text, DateTime reference)
        {
            var results = new List<DateTimeEntity>();

            if (string.IsNullOrEmpty(text)) return results;

            Parse(text.AsSpan(), reference, results);
            return results;
        }

        /// <summary>Scans <paramref name="text"/> and appends every recognised expression to <paramref name="results"/>.</summary>
        public void Parse(ReadOnlySpan<char> text, DateTime reference, List<DateTimeEntity> results)
        {
            if (text.IsEmpty) return;

            int capacity = Math.Min(Lexer.MaxLexemes, Math.Max(16, text.Length));

            var lexemes = ArrayPool<Lexeme>.Shared.Rent(capacity);
            var nodes   = ArrayPool<Node>.Shared.Rent(DefaultNodeCapacity);

            try
            {
                int count = Lexer.Tokenize(text, _lexicon, lexemes.AsSpan(0, capacity));
                if (count == 0) return;

                // Created on the first hit, so a scan that finds nothing allocates nothing at all
                Resolver resolver = null;

                int i = 0;

                while (i < count)
                {
                    var parser = new Parser(text, lexemes.AsSpan(0, count), _lexicon, nodes.AsSpan(0, DefaultNodeCapacity));

                    int end = parser.TryMatch(i, out int node);

                    if (end > i && node >= 0)
                    {
                        int charStart = nodes[node].CharStart;
                        int charEnd   = nodes[node].CharEnd;

                        Trim(text, ref charStart, ref charEnd);

                        if (charEnd > charStart)
                        {
                            resolver ??= new Resolver(nodes, reference, _lexicon);

                            var entity = resolver.Resolve(node, text.Slice(charStart, charEnd - charStart).ToString());

                            if (entity is object)
                            {
                                entity.Start = charStart;
                                entity.End   = charEnd - 1;
                                results.Add(entity);
                            }
                        }

                        i = end;
                    }
                    else
                    {
                        i++;
                    }
                }
            }
            finally
            {
                ArrayPool<Lexeme>.Shared.Return(lexemes, clearArray: true);
                ArrayPool<Node>.Shared.Return(nodes, clearArray: true);
            }
        }

        private static void Trim(ReadOnlySpan<char> text, ref int start, ref int end)
        {
            // "this week's" is reported as "this week"
            if (end - start > 2 && (text[end - 1] == 's' || text[end - 1] == 'S') && (text[end - 2] == '\'' || text[end - 2] == '\u2019')) end -= 2;

            while (end > start && IsTrimmable(text[end - 1])) end--;
            while (start < end && IsTrimmable(text[start]))   start++;
        }

        private static bool IsTrimmable(char c) => c is ' ' or '\t' or '\n' or '\r' or ',' or ';';
    }
}
