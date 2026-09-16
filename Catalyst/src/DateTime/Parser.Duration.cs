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
            n.DurationIsDateOnly = parts.IsDateOnly;
            n.Mod                = mod;
            SetSpan(ref n);

            node = Alloc(n);
            return at;
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

            if (!AtTerm(at, TermKind.Unit, out int unitValue))
            {
                if (business && amount >= 0) return -1;
                return -1;
            }

            at++;

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

            if (!AtWord(i, "and")) return -1;

            int at = i + 1;
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

                if (At(i + 1, LexKind.Dot) && AtNumber(i + 2) && !_lex[i + 1].SpaceBefore && !_lex[i + 2].SpaceBefore)
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
