using System;
using System.Runtime.CompilerServices;

namespace Catalyst.DateTimeRecognition
{
    public enum LexKind : byte
    {
        End = 0,
        Number,
        Word,
        Colon,
        Slash,
        Dash,
        Dot,
        Comma,
        LParen,
        RParen,
        Tilde,
        At,
        Equal,
        Less,
        Greater,
        Plus,
        Semicolon,
        Other,
    }

    /// <summary>
    /// One scanned unit of the input. Everything downstream works on these, never on the raw characters,
    /// so the grammar never needs to backtrack over text.
    /// </summary>
    public struct Lexeme
    {
        public int      Start;
        public int      Length;
        public LexKind  Kind;
        public TermInfo Term;
        /// <summary>Numeric value for <see cref="LexKind.Number"/>, clamped at <see cref="int.MaxValue"/>.</summary>
        public int      Number;
        /// <summary>Number of digits, so "05" and "2016" can be told apart from "5".</summary>
        public byte     Digits;
        public bool     SpaceBefore;
        /// <summary>How many lexemes a matched multi-word phrase covers, 1 when the term is a single word.</summary>
        public byte     PhraseLength;

        public readonly int End => Start + Length;
    }

    /// <summary>
    /// Character scanner. Splits on script changes (so "3pm" and "mar3" become two lexemes that the
    /// grammar can glue back together), resolves each word against the lexicon, and then folds
    /// multi-word phrases in a second pass. Writes into a caller-owned buffer and allocates nothing.
    /// </summary>
    public static class Lexer
    {
        public const int MaxLexemes = 4096;

        public static int Tokenize(ReadOnlySpan<char> text, Lexicon lexicon, Span<Lexeme> buffer)
        {
            int count = 0;
            int i     = 0;
            bool spaceBefore = false;

            while (i < text.Length && count < buffer.Length)
            {
                char c = text[i];

                if (IsSpace(c))
                {
                    spaceBefore = true;
                    i++;
                    continue;
                }

                int start = i;

                if (char.IsAsciiDigit(c))
                {
                    long value  = 0;
                    int  digits = 0;

                    while (i < text.Length && char.IsAsciiDigit(text[i]))
                    {
                        if (value < int.MaxValue) { value = value * 10 + (text[i] - '0'); }
                        digits++;
                        i++;
                    }

                    buffer[count++] = new Lexeme
                    {
                        Start        = start,
                        Length       = i - start,
                        Kind         = LexKind.Number,
                        Number       = (int)Math.Min(value, int.MaxValue),
                        Digits       = (byte)Math.Min(digits, 255),
                        SpaceBefore  = spaceBefore,
                        PhraseLength = 1,
                    };

                    spaceBefore = false;
                    continue;
                }

                // "'s ochtends", "'t", "o'clock" — a word may open on its apostrophe
                bool opensOnApostrophe = IsApostrophe(c) && i + 1 < text.Length && IsWordChar(text[i + 1])
                                         && (i == 0 || IsSpace(text[i - 1]));

                if (IsWordChar(c) || opensOnApostrophe)
                {
                    i++;

                    while (i < text.Length && (IsWordChar(text[i]) || (IsApostrophe(text[i]) && i + 1 < text.Length && IsWordChar(text[i + 1]))))
                    {
                        i++;
                    }

                    // A trailing apostrophe belongs to the word ("international workers' day")
                    if (i < text.Length && IsApostrophe(text[i]) && !(i + 1 < text.Length && char.IsAsciiDigit(text[i + 1])))
                    {
                        i++;
                    }

                    var word = text.Slice(start, i - start);

                    lexicon.TryGetWord(word, out var info);

                    // "neunundzwanzig" is one number written from its units and tens
                    if (info.Kind == TermKind.None && lexicon.TrySplitNumber(word, out var composed)) info = composed;

                    if (info.Kind == TermKind.None
                        && lexicon.TrySplitCompound(word, out int cut, out var head, out var tail)
                        && count + 1 < buffer.Length)
                    {
                        // "dienstagmorgen" is two words written as one
                        buffer[count++] = new Lexeme
                        {
                            Start        = start,
                            Length       = cut,
                            Kind         = LexKind.Word,
                            Term         = head,
                            SpaceBefore  = spaceBefore,
                            PhraseLength = 1,
                        };

                        buffer[count++] = new Lexeme
                        {
                            Start        = start + cut,
                            Length       = word.Length - cut,
                            Kind         = LexKind.Word,
                            Term         = tail,
                            SpaceBefore  = false,
                            PhraseLength = 1,
                        };

                        spaceBefore = false;
                        continue;
                    }

                    buffer[count++] = new Lexeme
                    {
                        Start        = start,
                        Length       = i - start,
                        Kind         = LexKind.Word,
                        Term         = info,
                        SpaceBefore  = spaceBefore,
                        PhraseLength = 1,
                    };

                    spaceBefore = false;
                    continue;
                }

                if (c == '½') // ½
                {
                    i++;
                    buffer[count++] = new Lexeme
                    {
                        Start        = start,
                        Length       = 1,
                        Kind         = LexKind.Word,
                        Term         = new TermInfo(TermKind.HalfWord),
                        SpaceBefore  = spaceBefore,
                        PhraseLength = 1,
                    };
                    spaceBefore = false;
                    continue;
                }

                i++;

                buffer[count++] = new Lexeme
                {
                    Start        = start,
                    Length       = 1,
                    Kind         = Classify(c),
                    SpaceBefore  = spaceBefore,
                    PhraseLength = 1,
                };

                spaceBefore = false;
            }

            FoldPhrases(text, lexicon, buffer.Slice(0, count));

            return count;
        }

        private static void FoldPhrases(ReadOnlySpan<char> text, Lexicon lexicon, Span<Lexeme> lexemes)
        {
            for (int i = 0; i < lexemes.Length; i++)
            {
                if (lexemes[i].Kind != LexKind.Word) continue;

                var first = text.Slice(lexemes[i].Start, lexemes[i].Length);

                if (!lexicon.TryGetPhrases(first, out var candidates)) continue;

                foreach (var phrase in candidates)
                {
                    int need    = phrase.Words.Length;
                    int matched = 1;
                    int at      = i + 1;

                    while (matched < need && at < lexemes.Length)
                    {
                        ref var lx = ref lexemes[at];

                        // Punctuation inside a phrase is skipped: "st. patrick's day" == "st patrick's day"
                        if (lx.Kind == LexKind.Dot || lx.Kind == LexKind.Dash || lx.Kind == LexKind.Comma)
                        {
                            at++;
                            continue;
                        }

                        if (lx.Kind != LexKind.Word) break;

                        if (!text.Slice(lx.Start, lx.Length).Equals(phrase.Words[matched], StringComparison.OrdinalIgnoreCase)) break;

                        matched++;
                        at++;
                    }

                    if (matched == need)
                    {
                        lexemes[i].Term         = phrase.Info;
                        lexemes[i].PhraseLength = (byte)Math.Min(at - i, 255);
                        break;
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsSpace(char c) => c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ' ' || c == '​';

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsApostrophe(char c) => c == '\'' || c == '’' || c == 'ʼ';

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsWordChar(char c) => char.IsLetter(c) || c == '_';

        private static LexKind Classify(char c) => c switch
        {
            ':'                          => LexKind.Colon,
            '/'                          => LexKind.Slash,
            '-' or '–' or '—'  => LexKind.Dash,
            '.'                          => LexKind.Dot,
            ','                          => LexKind.Comma,
            '('                          => LexKind.LParen,
            ')'                          => LexKind.RParen,
            '~'                          => LexKind.Tilde,
            '@'                          => LexKind.At,
            '='                          => LexKind.Equal,
            '<'                          => LexKind.Less,
            '>'                          => LexKind.Greater,
            '+'                          => LexKind.Plus,
            ';'                          => LexKind.Semicolon,
            _                            => LexKind.Other,
        };
    }
}
