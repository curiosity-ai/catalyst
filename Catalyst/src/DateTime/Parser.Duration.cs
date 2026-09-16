using System;
using System.Globalization;

namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        internal static string Fmt(double v) => DurationParts.Fmt(v);

        /// <summary>Reads a duration: "2 days", "3h", "one and a half hours", "more than 1 hour and 30 minutes".</summary>
        private int TryDuration(int i, out int node)
        {
            node = Node.Unspecified;

            int start = i;
            var mod   = ModKind.None;

            if (AtTerm(i, TermKind.Mod, out int modValue))
            {
                var k = (ModKind)modValue;

                if (k == ModKind.Less || k == ModKind.More || k == ModKind.Approx)
                {
                    mod = k;
                    i   = After(i);
                }
            }
            else if (AtTerm(i, TermKind.Approx))
            {
                mod = ModKind.Approx;
                i   = After(i);
            }

            // "a ½ hour" is half an hour; the article is not part of the fraction
            if ((AtWord(i, "a") || AtWord(i, "an")) && AtTerm(i + 1, TermKind.HalfWord))
            {
                i     = i + 1;
                start = i;
            }

            var parts = new DurationParts();
            int at    = i;
            bool any  = false;

            while (true)
            {
                int next = TryDurationComponent(at, ref parts);
                if (next < 0) break;

                any = true;
                at  = next;

                int sep = at;
                if (At(sep, LexKind.Comma)) sep++;
                if (AtWord(sep, "and")) sep++;

                if (sep != at)
                {
                    // Only consume the separator if another component really follows
                    var probe = parts;
                    if (TryDurationComponent(sep, ref probe) > 0) { at = sep; }
                    else break;
                }
            }

            if (!any || !parts.Any) return -1;

            var n = Node.Create(NodeKind.Duration);
            n.LexStart           = start;
            n.LexEnd             = at;
            n.Duration           = parts;
            n.DurationSeconds    = parts.TotalSeconds;
            n.DurationTimex      = parts.ToTimex();
            n.Mod                = mod;
            SetSpan(ref n);

            node = Alloc(n);
            return at;
        }

        /// <summary>
        /// The article in front of a unit that counts as one, outside English: a word that is both glue and
        /// the number one ("een", "una"), or the word for "whole" ("hele", "ganzen"), with at most one more
        /// piece of glue between it and the unit. Nothing but a unit may follow, so an English preposition —
        /// "for", "of", "during" — can never open a duration this way.
        /// </summary>
        private readonly int SkipDurationOne(int i)
        {
            // A preposition cannot head one: "por una hora" is reported from the article
            if (!AtTerm(i, TermKind.Article) && !AtTerm(i, TermKind.Whole) && !AtTermValue(i, TermKind.Cardinal, 1)) return i;

            int  at  = i;
            bool one = false;

            for (int k = 0; k < 3 && AtTerm(at, TermKind.Filler) && !AtTerm(at, TermKind.Unit); k++)
            {
                if (AtTerm(at, TermKind.Whole) || AtTermValue(at, TermKind.Cardinal, 1)) one = true;
                at = After(at);
            }

            return one && at > i && AtTerm(at, TermKind.Unit) ? at : i;
        }

        private readonly int LengthOfPhrase(int i)
        {
            int last = i + (_lex[i].PhraseLength < 1 ? 1 : _lex[i].PhraseLength) - 1;
            if (!In(last)) last = i;
            return _lex[last].End - _lex[i].Start;
        }

        /// <summary>One "&lt;amount&gt; &lt;unit&gt;" pair, with the English half/quarter idioms folded in.</summary>
        private readonly int TryDurationComponent(int i, ref DurationParts parts)
        {
            int    at     = i;
            double amount = double.NaN;

            // "a" / "an" / "the" / "all" / "another"
            if (AtWord(at, "a") || AtWord(at, "an") || AtWord(at, "the") || AtWord(at, "all") || AtWord(at, "another") || AtWord(at, "any"))
            {
                int probe = at + 1;

                if (AtTerm(probe, TermKind.HalfWord))
                {
                    amount = 0.5;
                    at     = probe + 1;
                }
                else if (AtTerm(probe, TermKind.Several, out int sv))
                {
                    amount = sv;
                    at     = probe + 1;
                }
                else
                {
                    amount = 1;
                    at     = probe;
                }
            }
            else if (SkipDurationOne(at) > at)
            {
                // "een uur", "den ganzen Tag", "todo el día" — the article is what counts as one
                amount = 1;
                at     = SkipDurationOne(at);
            }
            else if (AtTerm(at, TermKind.Several, out int several))
            {
                amount = several;
                at++;
            }
            else if (AtTerm(at, TermKind.HalfWord))
            {
                amount = 0.5;
                at++;
                at     = SkipWords(at, "a", "an");   // "half an hour"
            }
            else if (TryDecimal(at, out double d, out int afterNumber))
            {
                amount = d;
                at     = afterNumber;

                // "one and a half", "one and half", "one and a quarter"
                int fraction = TryFractionSuffix(at, out double extra);
                if (fraction > 0)
                {
                    amount += extra;
                    at      = fraction;
                }
            }
            else
            {
                return -1;
            }

            if (double.IsNaN(amount)) return -1;

            // "business days" / "working days"
            bool business = false;
            if (AtTerm(at, TermKind.BusinessDay))
            {
                business = true;
                at++;
            }

            // "eine Viertelstunde", "ein Dreiviertelstunde" — a fraction between the number and the unit
            if (!double.IsNaN(amount) && AtTerm(at, TermKind.Unit) == false)
            {
                if (AtTerm(at, TermKind.HalfWord))                     { amount *= 0.5; at = After(at); }
                else if (AtTerm(at, TermKind.QuarterWord, out int qw)) { amount *= qw == 3 ? 0.75 : 0.25; at = After(at); }
            }

            if (!AtTerm(at, TermKind.Unit, out int unitValue)) return -1;

            // "the second week of 2021" counts weeks; "second" is the ordinal, not the unit
            if (_lex[at].Term.Kind == TermKind.Ordinal && AtTerm(at + 1, TermKind.Unit)) return -1;

            at = After(at);

            var unit = (TimeUnit)unitValue;
            if (business && unit == TimeUnit.Day) unit = TimeUnit.BusinessDay;

            // "one hour and a half" / "one year and a quarter"
            int trailing = TryFractionSuffix(at, out double trailingExtra);
            if (trailing > 0)
            {
                amount += trailingExtra;
                at      = trailing;
            }

            parts.Add(unit, amount);
            return at;
        }

        /// <summary>"and a half", "and half", "and a quarter" — returns the index past it, or -1.</summary>
        private readonly int TryFractionSuffix(int i, out double extra)
        {
            extra = 0;

            // "zweieinhalb Stunden" — the half written onto the number it follows needs no joiner
            if (In(i) && !_lex[i].SpaceBefore && AtTerm(i, TermKind.HalfWord))    { extra = 0.5;  return After(i); }
            if (In(i) && !_lex[i].SpaceBefore && AtTerm(i, TermKind.QuarterWord)) { extra = 0.25; return After(i); }

            if (!AtTerm(i, TermKind.AndWord)) return -1;

            int at = After(i);
            at = SkipWords(at, "a", "an");

            if (AtTerm(at, TermKind.HalfWord))
            {
                extra = 0.5;
                return at + 1;
            }

            if (AtTerm(at, TermKind.QuarterWord))
            {
                extra = 0.25;
                return at + 1;
            }

            return -1;
        }

        /// <summary>A number that may carry a decimal part written with a dot ("3.5years", "123.45 sec").</summary>
        private readonly bool TryDecimal(int i, out double value, out int end)
        {
            value = 0;
            end   = i;

            if (AtNumber(i))
            {
                value = NumberAt(i);
                end   = i + 1;

                if ((At(i + 1, LexKind.Dot) || (_lexicon.DecimalComma && At(i + 1, LexKind.Comma))) && AtNumber(i + 2) && !_lex[i + 1].SpaceBefore && !_lex[i + 2].SpaceBefore)
                {
                    int digits = DigitsAt(i + 2);
                    value += NumberAt(i + 2) / Math.Pow(10, digits);
                    end    = i + 3;
                }

                return true;
            }

            if (TryWordNumber(i, out int words, out int afterWords))
            {
                value = words;
                end   = afterWords;
                return true;
            }

            return false;
        }

        private readonly void SetSpan(ref Node n)
        {
            n.CharStart = _lex[n.LexStart].Start;
            n.CharEnd   = _lex[n.LexEnd - 1].End;
        }
    }
}
