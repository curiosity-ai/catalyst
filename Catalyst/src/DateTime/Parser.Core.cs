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

        public int Count => _lex.Length;

        public Span<Node> Nodes => _nodes.Slice(0, _nodeCount);

        public void Reset() => _nodeCount = 0;

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

        private readonly int SkipComma(int i) => At(i, LexKind.Comma) ? i + 1 : i;

        private readonly bool IsRangeConnector(int i)
        {
            if (AtTerm(i, TermKind.Connector)) return true;
            return At(i, LexKind.Dash) || At(i, LexKind.Tilde);
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

            while (In(at))
            {
                if (AtTerm(at, TermKind.Cardinal, out int v))
                {
                    if      (current == 0)                                                { current = v; }
                    else if (current >= 100 && current % 100 == 0 && v < 100 && v >= 10)  { current += v; }
                    else if (current % 10 == 0 && v < 10)                                 { current += v; }
                    else                                                                  { break; }

                    any = true;
                    at++;
                    continue;
                }

                if (AtTerm(at, TermKind.Multiplier, out int m) && any)
                {
                    if (m >= 1000)
                    {
                        total  += (current == 0 ? 1 : current) * m;
                        current = 0;
                    }
                    else
                    {
                        current = (current == 0 ? 1 : current) * m;
                    }

                    at++;
                    continue;
                }

                // "two thousand and fifteen" / "twenty-five"
                if (any && (At(at, LexKind.Dash) || AtWord(at, "and")) && In(at + 1) && (AtTerm(at + 1, TermKind.Cardinal) || AtTerm(at + 1, TermKind.Multiplier)))
                {
                    at++;
                    continue;
                }

                break;
            }

            value = total + current;
            end   = at;
            return any;
        }

        /// <summary>An ordinal, as "1st" / "21st" / "first" / "twenty third" / "thirty-first".</summary>
        private readonly bool TryOrdinal(int i, out int value, out int end)
        {
            value = 0;
            end   = i;

            if (AtNumber(i) && AtTerm(i + 1, TermKind.OrdinalSuffix) && !_lex[i + 1].SpaceBefore)
            {
                value = NumberAt(i);
                end   = i + 2;
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

        /// <summary>An ordinal that may be introduced by "the".</summary>
        private readonly bool TryOrdinalWithArticle(int i, out int value, out int end)
        {
            int at = SkipWord(i, "the");

            if (TryOrdinal(at, out value, out end)) return true;

            value = 0;
            end   = i;
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

        // ------------------------------------------------------------------ modifiers

        /// <summary>Reads an optional leading modifier ("before", "after", "around", "early", "the end of", ...).</summary>
        private readonly bool TryLeadingMod(int i, out ModKind mod, out int end)
        {
            mod = ModKind.None;
            end = i;

            if (AtTerm(i, TermKind.Approx))
            {
                mod = ModKind.Approx;
                end = After(i);
                return true;
            }

            if (AtTerm(i, TermKind.Mod, out int value))
            {
                var k = (ModKind)value;

                if (k == ModKind.OrLater || k == ModKind.OrEarlier) return false;

                mod = k;
                end = After(i);
                return true;
            }

            return false;
        }
    }
}
