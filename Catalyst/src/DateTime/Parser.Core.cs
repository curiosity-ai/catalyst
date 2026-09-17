using System;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>
    /// The grammar. Every construct is matched by walking the lexeme array forward — there is no
    /// backtracking over characters and no regular expression anywhere in the engine.
    /// </summary>
    public ref partial struct Parser
    {
        private readonly ReadOnlySpan<char>   _text;
        private readonly ReadOnlySpan<Lexeme> _lex;
        private readonly Lexicon              _lexicon;
        private          Span<Node>           _nodes;
        private          int                  _nodeCount;
        private          int                  _modDepth;

        public Parser(ReadOnlySpan<char> text, ReadOnlySpan<Lexeme> lexemes, Lexicon lexicon, Span<Node> nodes)
        {
            _text      = text;
            _lex       = lexemes;
            _lexicon   = lexicon;
            _nodes     = nodes;
            _nodeCount = 0;
            _modDepth  = 0;
        }

        public int Alloc(Node node)
        {
            if (_nodeCount >= _nodes.Length) return Node.Unspecified;
            _nodes[_nodeCount] = node;
            return _nodeCount++;
        }

        public ref Node NodeAt(int index) => ref _nodes[index];

        // ------------------------------------------------------------------ lexeme helpers

        private readonly bool In(int i) => (uint)i < (uint)_lex.Length;

        private readonly LexKind KindOf(int i) => In(i) ? _lex[i].Kind : LexKind.End;

        private readonly bool At(int i, LexKind kind) => In(i) && _lex[i].Kind == kind;

        private readonly bool AtTerm(int i, TermKind kind) => In(i) && _lex[i].Kind == LexKind.Word && _lex[i].Term.Is(kind);

        private readonly bool AtTerm(int i, TermKind kind, out int value)
        {
            if (In(i) && _lex[i].Kind == LexKind.Word && _lex[i].Term.Is(kind, out value)) return true;
            value = 0;
            return false;
        }

        private readonly bool AtTermValue(int i, TermKind kind, int value) => AtTerm(i, kind, out var v) && v == value;

        /// <summary>The index just past the term at <paramref name="i"/>, accounting for multi-word phrases.</summary>
        private readonly int After(int i) => In(i) ? i + (_lex[i].PhraseLength < 1 ? 1 : _lex[i].PhraseLength) : i + 1;

        private readonly bool AtNumber(int i) => At(i, LexKind.Number);

        private readonly int NumberAt(int i) => In(i) ? _lex[i].Number : 0;

        private readonly int DigitsAt(int i) => In(i) ? _lex[i].Digits : 0;

        private readonly bool AtWord(int i, string word) => In(i) && _lex[i].Kind == LexKind.Word && _text.Slice(_lex[i].Start, _lex[i].Length).Equals(word, StringComparison.OrdinalIgnoreCase);

        /// <summary>Skips a specific filler word if it is at <paramref name="i"/>.</summary>
        private readonly int SkipWord(int i, string word) => AtWord(i, word) ? i + 1 : i;

        private readonly int SkipWords(int i, string a, string b)
        {
            i = SkipWord(i, a);
            i = SkipWord(i, b);
            return i;
        }

        /// <summary>
        /// Skips a leading definite article, and says whether it should be left outside the match. A date
        /// keeps its article in English and drops it everywhere else; a period is the other way round.
        /// </summary>
        private readonly int SkipArticleOfDate(int i, out int spanStart)
        {
            int at    = SkipArticle(i);
            spanStart = _lexicon.ArticleInDateSpan ? i : at;
            return at;
        }

        /// <summary>
        /// Skips a leading definite article. English reports "next week" without its "the", so there the
        /// article is only stepped over; the other languages keep theirs inside the match.
        /// </summary>
        /// <summary>A word that introduces a clock reading: "at 5", "a las 5", "um 8 Uhr".</summary>
        private readonly bool AtClockPrefix(int i) => AtTerm(i, TermKind.ClockPrefix);

        private readonly int SkipClockPrefix(int i) => AtClockPrefix(i) ? After(i) : i;

        /// <summary>Whether a clock introducer ends where <paramref name="i"/> begins, phrase or single word.</summary>
        private readonly bool ClockPrefixEndsAt(int i)
        {
            for (int k = i - 1; k >= 0 && k >= i - 3; k--)
            {
                if (AtClockPrefix(k) && After(k) == i) return true;
            }

            return false;
        }

        /// <summary>Whether the word at <paramref name="i"/> is written as a plural. Dutch names its parts
        /// of the day with a trailing -s that is not one, so a language that does not mark plurals this
        /// way never answers true.</summary>
        private readonly bool ShowsPlural(int i)
        {
            if (!In(i)) return false;

            // German marks the recurring weekday with an -s it does not use for plurals elsewhere:
            // "montags" is every Monday, and "Montag" being a word of its own is what says so
            if (!_lexicon.PluralEndsInS && !AtTerm(i, TermKind.Weekday)) return false;

            return _lexicon.IsPluralOfKnownWord(_text.Slice(_lex[i].Start, _lex[i].Length));
        }

        /// <summary>Whether the word at <paramref name="i"/> reads as a plural, where the language shows it.</summary>
        private readonly bool LooksPlural(int i)
        {
            if (!_lexicon.PluralEndsInS) return true;
            if (!In(i)) return false;

            var word = _text.Slice(_lex[i].Start, _lex[i].Length);
            return word.Length > 1 && (word[^1] == 's' || word[^1] == 'S');
        }

        /// <summary>
        /// Whether the word at <paramref name="i"/> is an article or a bare preposition — glue that counts
        /// as "one" in front of a unit ("a day", "una hora", "een uur", "ganzen Tag").
        /// </summary>
        private readonly bool AtArticle(int i)
        {
            if (!AtTerm(i, TermKind.Filler)) return false;

            return !AtTerm(i, TermKind.Month)       && !AtTerm(i, TermKind.Weekday)  && !AtTerm(i, TermKind.Unit)
                && !AtTerm(i, TermKind.Relative)    && !AtTerm(i, TermKind.SpecialDay) && !AtTerm(i, TermKind.Cardinal)
                && !AtTerm(i, TermKind.ClockPrefix) && !AtTerm(i, TermKind.InPrefix);
        }

        private readonly int SkipArticle(int i)
        {
            if (AtWord(i, "the")) return i + 1;

            if ((!_lexicon.ArticleInDateSpan || _lexicon.ArticleInPeriodSpan)
                && AtTerm(i, TermKind.Filler)
                && !AtTerm(i, TermKind.Month) && !AtTerm(i, TermKind.Weekday) && !AtTerm(i, TermKind.Unit)
                && !AtTerm(i, TermKind.Relative) && !AtTerm(i, TermKind.SpecialDay) && !AtTerm(i, TermKind.Cardinal))
            {
                return i + 1;
            }

            return i;
        }

        /// <summary>
        /// Skips the small connective words a date is written with — English "of"/"the", Spanish "de"/"del",
        /// German "im"/"am", … . They are all <see cref="TermKind.Filler"/>, so the grammar stays language-neutral.
        /// </summary>
        private readonly int SkipGlue(int i, int max = 2)
        {
            int at = i;

            for (int k = 0; k < max && AtTerm(at, TermKind.Filler) && !AtTerm(at, TermKind.Month) && !AtTerm(at, TermKind.Weekday) && !AtTerm(at, TermKind.Unit); k++)
            {
                at++;
            }

            return at;
        }

        // ------------------------------------------------------------------ numbers

        /// <summary>A plain integer, written either as digits or as English words.</summary>
        private readonly bool TryInteger(int i, out int value, out int end)
        {
            if (AtNumber(i))
            {
                value = NumberAt(i);
                end   = i + 1;
                return true;
            }

            return TryWordNumber(i, out value, out end);
        }

        /// <summary>One standard English number group: "twenty five", "two thousand and fifteen", "seventeen hundred thirty three".</summary>
        private readonly bool TryWordNumber(int i, out int value, out int end)
        {
            int total   = 0;
            int current = 0;
            bool any    = false;
            int at      = i;

            // Where the number stops, which is not where the scan stops: "otto e un quarto" reads eight and
            // steps over its joiner before finding that "un" cannot go on it, and the joiner is not the number's
            int lastGood = i;

            while (In(at))
            {
                if (AtTerm(at, TermKind.Cardinal, out int v))
                {
                    if      (current == 0)                                                { current = v; }
                    else if (current >= 100 && current % 100 == 0 && v < 100 && v >= 10)  { current += v; }
                    else if (current % 10 == 0 && v < 10)                                 { current += v; }
                    // "soixante-dix", "quatre-vingt-dix" — a vigesimal ten counts its teens on
                    else if ((current == 60 || current == 80) && v >= 10 && v < 20)       { current += v; }
                    else                                                                  { break; }

                    any      = true;
                    at       = After(at);   // "quatre-vingt" is one cardinal written as a phrase
                    lastGood = at;
                    continue;
                }

                // "mil novecientos noventa y dos" opens on its thousand, where English says "one thousand"
                if (AtTerm(at, TermKind.Multiplier, out int m) && (any || m >= 1000))
                {
                    any = true;

                    if (m >= 1000)
                    {
                        total  += (current == 0 ? 1 : current) * m;
                        current = 0;
                    }
                    else
                    {
                        current = (current == 0 ? 1 : current) * m;
                    }

                    at       = After(at);
                    lastGood = at;
                    continue;
                }

                // "cuarenta y dos" / "vierzig und zwei" — tens and units joined by the language's "and"
                bool joinsUnits = AtTerm(at, TermKind.Connector) && !AtTerm(at, TermKind.ToWord)
                                  && current >= 20 && current % 10 == 0
                                  && AtTerm(at + 1, TermKind.Cardinal, out int units) && units < 10;

                // "two thousand and fifteen" / "twenty-five"
                if (any && (At(at, LexKind.Dash) || AtTerm(at, TermKind.AndWord) || joinsUnits) && In(at + 1) && (AtTerm(at + 1, TermKind.Cardinal) || AtTerm(at + 1, TermKind.Multiplier)))
                {
                    at++;
                    continue;
                }

                break;
            }

            value = total + current;
            end   = lastGood;
            return any;
        }

        /// <summary>An ordinal, as "1st" / "21st" / "first" / "twenty third" / "thirty-first".</summary>
        /// <summary>
        /// Steps over the word for "day" when it introduces a day of the month — "el día 21", "dia 12",
        /// "o dia 4". It does for those languages what "the" does in "the 21st", so the number that follows
        /// reads as a date rather than as a bare number.
        /// </summary>
        private readonly int SkipDayNoun(int i)
        {
            int at = SkipArticle(i);

            if (!AtTermValue(at, TermKind.Unit, (int)TimeUnit.Day)) return i;

            at = After(at);

            if (AtNumber(at)) return DigitsAt(at) <= 2 && NumberAt(at) >= 1 && NumberAt(at) <= 31 ? at : i;

            // "dia vinte e cinco" — the number it introduces may be spelled out
            return TryWordNumber(at, out int spoken, out _) && spoken >= 1 && spoken <= 31 ? at : i;
        }

        /// <summary>"22." — where the language writes an ordinal as its number and a full stop.</summary>
        private readonly bool AtDottedOrdinal(int i)
            => _lexicon.OrdinalEndsInDot && AtNumber(i) && At(i + 1, LexKind.Dot) && !_lex[i + 1].SpaceBefore;

        /// <summary>Whether the word at <paramref name="i"/> is one an ordinal's number is written with.</summary>
        private readonly bool AtOrdinalSuffix(int i)
            => In(i) && _lex[i].Kind == LexKind.Word
            && (_lex[i].Term.Is(TermKind.OrdinalSuffix) || _lexicon.IsOrdinalSuffix(_text.Slice(_lex[i].Start, _lex[i].Length)));

        private readonly bool TryOrdinal(int i, out int value, out int end)
        {
            value = 0;
            end   = i;

            if (AtNumber(i) && (AtOrdinalSuffix(i + 1) || AtDottedOrdinal(i)) && !_lex[i + 1].SpaceBefore)
            {
                value = NumberAt(i);
                end   = i + 2;
                return true;
            }

            // "1.º" — the suffix written after a full stop
            if (AtNumber(i) && At(i + 1, LexKind.Dot) && !_lex[i + 1].SpaceBefore
                && AtOrdinalSuffix(i + 2) && !_lex[i + 2].SpaceBefore)
            {
                value = NumberAt(i);
                end   = i + 3;
                return true;
            }

            // "twenty third", "thirty-first"
            if (AtTerm(i, TermKind.Cardinal, out int tens) && tens >= 20 && tens % 10 == 0)
            {
                int at = i + 1;
                if (At(at, LexKind.Dash)) at++;

                if (AtTerm(at, TermKind.Ordinal, out int ones) && ones < 10)
                {
                    value = tens + ones;
                    end   = at + 1;
                    return true;
                }
            }

            if (AtTerm(i, TermKind.Ordinal, out int single))
            {
                value = single;
                end   = After(i);
                return true;
            }

            return false;
        }

        /// <summary>A four-digit year, or one spelled out ("nineteen seventy two", "two thousand and fifteen").</summary>
        private readonly bool TryYear(int i, out int year, out int end)
        {
            year = Node.Unspecified;
            end  = i;

            if (AtNumber(i) && DigitsAt(i) == 4 && NumberAt(i) >= 1000 && NumberAt(i) <= 3000)
            {
                year = NumberAt(i);
                end  = i + 1;
                return true;
            }

            if (TryWordNumber(i, out int first, out int afterFirst))
            {
                if (first >= 1000 && first <= 3000)
                {
                    year = first;
                    end  = afterFirst;
                    return true;
                }

                if (first >= 10 && first <= 30 && TryWordNumber(afterFirst, out int second, out int afterSecond) && second >= 0 && second <= 99)
                {
                    year = first * 100 + second;
                    end  = afterSecond;
                    return true;
                }
            }

            return false;
        }

        /// <summary>A two-digit year, expanded the way Microsoft.Recognizers.Text does: 00-30 -> 2000s, 31-99 -> 1900s.</summary>
        public static int ExpandTwoDigitYear(int y) => y < 100 ? (y < 30 ? 2000 + y : 1900 + y) : y;
    }
}
