using System;

namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        /// <summary>The clock window a part of the day covers, and the TIMEX letters that stand for it.</summary>
        public readonly struct DayPart
        {
            public readonly string Timex;
            public readonly int    StartHour;
            public readonly int    StartMinute;
            public readonly int    EndHour;
            public readonly int    EndMinute;
            public readonly int    EndSecond;

            public DayPart(string timex, int startHour, int endHour, int startMinute = 0, int endMinute = 0, int endSecond = 0)
            {
                Timex       = timex;
                StartHour   = startHour;
                EndHour     = endHour;
                StartMinute = startMinute;
                EndMinute   = endMinute;
                EndSecond   = endSecond;
            }
        }

        public static DayPart RangeOf(PartOfDayKind kind) => kind switch
        {
            PartOfDayKind.Morning      => new DayPart("TMO",   8, 12),
            PartOfDayKind.Afternoon    => new DayPart("TAF",  12, 16),
            PartOfDayKind.Evening      => new DayPart("TEV",  16, 20),
            PartOfDayKind.Night        => new DayPart("TNI",  20, 23, 0, 59, 59),
            PartOfDayKind.Tonight      => new DayPart("TNI",  20, 23, 0, 59, 59),
            PartOfDayKind.DayTime      => new DayPart("TDT",   8, 18),
            PartOfDayKind.LateNight    => new DayPart("TNT",   0,  8),
            PartOfDayKind.Business     => new DayPart("TBH",   8, 18),
            PartOfDayKind.Lunch        => new DayPart("TMEL", 11, 13),
            PartOfDayKind.Dinner       => new DayPart("TMED", 16, 20),
            PartOfDayKind.Breakfast    => new DayPart("TMEB",  8, 12),
            PartOfDayKind.Brunch       => new DayPart("TMEBR",10, 12),
            PartOfDayKind.EarlyMorning => new DayPart("TMO",   8, 10),
            _                          => new DayPart(null,    0,  0),
        };

        // ------------------------------------------------------------------ am / pm

        /// <summary>"am", "pm", "a.m.", "p . m .", or a bare "a"/"p" glued to the clock.</summary>
        private readonly bool TryAmPm(int i, out int value, out int end)
        {
            value = Node.Unspecified;
            end   = i;

            if (!At(i, LexKind.Word)) return false;

            // "de la tarde", "du matin" — a language can name the half of the day instead of writing am/pm.
            // A word that is also a part of the day keeps that reading, which carries more than the half.
            if (AtTerm(i, TermKind.AmPm, out int named) && _lex[i].PhraseLength > 1
                && !AtTerm(i, TermKind.PartOfDay) && !AtTerm(i, TermKind.Filler))
            {
                value = named;
                end   = After(i);
                return true;
            }

            var word = _text.Slice(_lex[i].Start, _lex[i].Length);

            if (word.Equals("am", StringComparison.OrdinalIgnoreCase) || word.Equals("pm", StringComparison.OrdinalIgnoreCase))
            {
                value = word[0] is 'a' or 'A' ? 0 : 1;
                end   = i + 1;
                return true;
            }

            bool isA = word.Equals("a", StringComparison.OrdinalIgnoreCase);
            bool isP = word.Equals("p", StringComparison.OrdinalIgnoreCase);

            if (!isA && !isP) return false;

            // "a . m ." / "p.m."
            int at      = i + 1;
            bool dotted = At(at, LexKind.Dot);
            if (dotted) at++;

            if (AtWord(at, "m"))
            {
                at++;

                // "p.m." and "p . m ." end in a dot; the dot in "one thirty p m." ends the sentence
                if (At(at, LexKind.Dot) && (dotted || _lex[at].SpaceBefore)) at++;

                value = isA ? 0 : 1;
                end   = at;
                return true;
            }

            // A bare "a"/"p" only counts when it is glued to what precedes it ("9:00a", "7p")
            if (!_lex[i].SpaceBefore)
            {
                value = isA ? 0 : 1;
                end   = i + 1;
                return true;
            }

            return false;
        }

        // ------------------------------------------------------------------ part of the day

        /// <summary>"in the morning", "early afternoon", "at lunchtime", "later in the evening".</summary>
        private readonly int TryPartOfDay(int i, out PartOfDayKind kind, out ModKind mod) => TryPartOfDay(i, out kind, out mod, out _);

        private readonly int TryPartOfDay(int i, out PartOfDayKind kind, out ModKind mod, out int spanStart)
        {
            kind      = PartOfDayKind.None;
            mod       = ModKind.None;
            spanStart = i;

            int at = i;

            // "in the morning", "dans la soiree", "am Nachmittag" — one or two connective words, whatever they are
            at = SkipGlue(at, 2);

            if (AtTerm(at, TermKind.Mod, out int modValue))
            {
                var k = (ModKind)modValue;
                if (k == ModKind.Early || k == ModKind.Late || k == ModKind.Mid)
                {
                    // "in late afternoon" is reported from the modifier, where "in the morning" keeps its lead-in
                    mod       = k;
                    spanStart = at;
                    at        = After(at);
                    at        = SkipWords(at, "in", "the");
                }
            }
            else if (AtTerm(at, TermKind.FromNow) && AtWord(at + 1, "in"))
            {
                // "later in the morning"
                mod = ModKind.Late;
                at  = at + 2;
                at  = SkipWord(at, "the");
            }

            if (At(at, LexKind.Dash) && at > i) at++;

            if (!AtTerm(at, TermKind.PartOfDay, out int podValue)) return -1;

            kind = (PartOfDayKind)podValue;
            at   = After(at);

            // "night-time", "day time"
            if ((At(at, LexKind.Dash) || AtWord(at, "time")) && kind == PartOfDayKind.Night)
            {
                int probe = At(at, LexKind.Dash) ? at + 1 : at;
                if (AtWord(probe, "time"))
                {
                    kind = PartOfDayKind.LateNight;
                    at   = probe + 1;
                }
            }

            if (kind == PartOfDayKind.Morning && mod == ModKind.Early) { /* keeps TMO with a start mod */ }

            return at;
        }

        // ------------------------------------------------------------------ the clock itself

        /// <summary>Reads a wall-clock reading and leaves the am/pm decision to the caller.</summary>
        private readonly int TryClock(int i, out int hour, out int minute, out int second, out int ampm, out bool explicitMinutes, out bool marked, out bool dottedMinutes)
        {
            hour            = Node.Unspecified;
            minute          = Node.Unspecified;
            second          = Node.Unspecified;
            ampm            = Node.Unspecified;
            explicitMinutes = false;
            marked          = false;
            dottedMinutes   = false;

            int at = i;

            // "noon" / "midnight" / "noonish", and "12 noon" / "12 midnight", where the hour repeats the word
            int spoken = AtNumber(at) && (NumberAt(at) == 12 || NumberAt(at) == 0) && AtTerm(at + 1, TermKind.PartOfDay) ? at + 1 : at;

            if (AtTerm(spoken, TermKind.PartOfDay, out int podValue))
            {
                var pod = (PartOfDayKind)podValue;

                if (pod == PartOfDayKind.Noon)     { hour = 12; ampm = 1; marked = true; return After(spoken); }
                if (pod == PartOfDayKind.Midnight) { hour = 0;  ampm = 0; marked = true; return After(spoken); }
            }

            // "halb acht", "viertel acht", "dreiviertel acht" — the fraction counts towards the hour it names
            if (_lexicon.HalfIsBeforeTheHour
                && (AtTerm(at, TermKind.HalfWord) || AtTerm(at, TermKind.QuarterWord, out _))
                && TryHourValue(After(at), out int fractionOf, out int afterFractionOf))
            {
                int quarters = AtTerm(at, TermKind.HalfWord) ? 2 : AtTermValue(at, TermKind.QuarterWord, 3) ? 3 : 1;

                hour            = fractionOf == 1 ? 12 : fractionOf - 1;
                minute          = quarters * 15;
                explicitMinutes = true;
                marked          = true;
                return afterFractionOf;
            }

            // "half past seven", "quarter to five", "ten past nine", "twenty minutes past eight"
            int relative = TryRelativeMinutes(at, out int relMinute, out int relDirection, out int relEnd);
            if (relative > 0)
            {
                if (!TryHourValue(relEnd, out int baseHour, out int afterBase)) return -1;

                hour            = baseHour;
                minute          = relDirection > 0 ? relMinute : 60 - relMinute;
                if (relDirection < 0) hour = hour == 1 ? 12 : hour - 1;
                second          = Node.Unspecified;
                explicitMinutes = true;

                int tail = afterBase;
                if (AtTerm(tail, TermKind.OClock)) tail = After(tail);

                marked = true;
                return tail;
            }

            // "1140 a.m." — a four-digit military reading
            if (AtNumber(i) && DigitsAt(i) == 4 && NumberAt(i) <= 2359 && (NumberAt(i) % 100) < 60)
            {
                int candidateHour   = NumberAt(i) / 100;
                int candidateMinute = NumberAt(i) % 100;

                if (candidateHour <= 23)
                {
                    int probe    = i + 1;
                    bool hasAmPm = TryAmPm(probe, out int ap, out int apEnd);

                    if (hasAmPm || IsInClockRangeContext(i))
                    {
                        hour            = candidateHour;
                        minute          = candidateMinute;
                        explicitMinutes = true;
                        marked          = true;
                        if (hasAmPm) { ampm = ap; return apEnd; }
                        return probe;
                    }
                }

                return -1;
            }

            if (!TryHourValue(at, out hour, out int afterHour)) return -1;

            at = afterHour;

            if (hour < 0 || hour > 24) return -1;

            // ":mm[:ss]" — a clock writes its minutes with two digits, so "1:1" and "4:3" are not times
            if (At(at, LexKind.Colon) && AtNumber(at + 1) && DigitsAt(at + 1) == 2 && NumberAt(at + 1) < 60)
            {
                minute          = NumberAt(at + 1);
                explicitMinutes = true;
                at              = at + 2;

                if (At(at, LexKind.Colon) && AtNumber(at + 1) && DigitsAt(at + 1) <= 2 && NumberAt(at + 1) < 60)
                {
                    second = NumberAt(at + 1);
                    at     = at + 2;
                }
            }
            // ".mm" — "8.10 pm", "at 6.45"
            else if (At(at, LexKind.Dot) && AtNumber(at + 1) && DigitsAt(at + 1) == 2 && NumberAt(at + 1) < 60 && !_lex[at].SpaceBefore && !_lex[at + 1].SpaceBefore)
            {
                minute          = NumberAt(at + 1);
                explicitMinutes = true;
                dottedMinutes   = true;
                at              = at + 2;
            }
            // spelled-out minutes — "three thirty", "two forty five", "siete y media", "dos cuarenta y dos"
            else if (AtTerm(i, TermKind.Cardinal) && TrySpokenMinutes(at, out int spokenMinutes, out int afterSpoken))
            {
                // "siete menos cuarto" counts backwards from the hour it just named
                if (spokenMinutes < 0)
                {
                    minute = 60 + spokenMinutes;
                    hour   = hour == 1 ? 12 : hour - 1;
                }
                else
                {
                    minute = spokenMinutes;
                }

                explicitMinutes = true;
                at              = afterSpoken;

                if (AtTermValue(at, TermKind.Unit, (int)TimeUnit.Minute)) at++;
            }

            if (AtTerm(at, TermKind.OClock) && IsClockMarker(at, i, hour))
            {
                at     = After(at);
                marked = true;

                // "12h00" / "10h30" — the minutes follow the marker
                if (minute < 0 && AtNumber(at) && DigitsAt(at) == 2 && NumberAt(at) < 60)
                {
                    minute          = NumberAt(at);
                    explicitMinutes = true;
                    at++;
                }
            }

            if (At(at, LexKind.Dot) && AtTerm(at + 1, TermKind.AmPm)) at++;   // "9.am"

            if (TryAmPm(at, out int ampmValue, out int ampmEnd))
            {
                ampm = ampmValue;
                at   = ampmEnd;
            }
            else if (AtTerm(at, TermKind.Approx) && !_lex[at].SpaceBefore)
            {
                at++;   // "11ish"
                marked = true;
            }

            // "7 en punto", "5 pm o'clock" — a trailing marker on a reading that is already a clock.
            // "tres horas" is a duration, so the hour unit alone does not make one.
            if (AtTerm(at, TermKind.OClock) && (ampm >= 0 || marked || explicitMinutes || IsClockMarker(at, i, hour)))
            {
                at     = After(at);
                marked = true;
            }

            return at;
        }

        /// <summary>
        /// The minutes spoken after the hour: "three thirty", and where the language joins them to it,
        /// "siete y media" and "dos cuarenta y dos".
        /// </summary>
        private readonly bool TrySpokenMinutes(int i, out int minutes, out int end)
        {
            minutes = Node.Unspecified;
            end     = i;

            int at = At(i, LexKind.Dash) ? i + 1 : i;

            // "siete menos cuarto", "sette meno un quarto" — the minutes come off the hour just read
            if (_lexicon.MinutesFollowHour && AtTerm(at, TermKind.ToWord) && !AtTerm(at, TermKind.Connector))
            {
                int back = SkipArticle(After(at));

                if (AtTerm(back, TermKind.QuarterWord)) { minutes = -15; end = After(back); return true; }
                if (AtTerm(back, TermKind.HalfWord))    { minutes = -30; end = After(back); return true; }

                if (TryWordNumber(back, out int off, out int afterOff) && off > 0 && off < 60)
                {
                    minutes = -off;
                    end     = afterOff;
                    return true;
                }

                return false;
            }

            if (_lexicon.MinutesFollowHour && AtTerm(at, TermKind.Connector) && !AtTerm(at, TermKind.ToWord))
            {
                int joined = After(at);

                if (AtTerm(joined, TermKind.HalfWord))         { minutes = 30; end = After(joined); return true; }
                if (AtTerm(joined, TermKind.QuarterWord))      { minutes = 15; end = After(joined); return true; }

                at = joined;
            }

            if (TryWordNumber(at, out int spoken, out int afterSpoken) && spoken > 0 && spoken < 60)
            {
                minutes = spoken;
                end     = afterSpoken;
                return true;
            }

            return false;
        }

        /// <summary>
        /// Whether the o'clock word at <paramref name="at"/> really marks a clock. In several languages the
        /// word is also the duration unit ("2 heures", "2 ore"), so it only reads as a clock when the reading
        /// cannot be a plain count: a value past noon, minutes after it, or a preposition in front.
        /// </summary>
        private readonly bool IsClockMarker(int at, int hourAt, int hour)
        {
            if (_lex[at].Term.Kind != TermKind.Unit) return true;

            if (hour > 12 && hour < 24) return true;   // "24 uur" is a day's worth, not midnight
            if (AtNumber(at + 1) && DigitsAt(at + 1) == 2 && NumberAt(at + 1) < 60) return true;

            int before = hourAt - 1;

            return AtTerm(before, TermKind.Connector) || AtTerm(before, TermKind.Approx)
                || AtTerm(before, TermKind.RangeStart) || AtTerm(before, TermKind.ClockPrefix);
        }

        /// <summary>An hour, as digits or spelled out.</summary>
        private readonly bool TryHourValue(int i, out int hour, out int end)
        {
            if (AtNumber(i) && DigitsAt(i) <= 2)
            {
                hour = NumberAt(i);
                end  = i + 1;
                return hour >= 0 && hour <= 24;
            }

            if (AtTerm(i, TermKind.Cardinal, out int spoken) && spoken >= 1 && spoken <= 24)
            {
                hour = spoken;
                end  = i + 1;
                return true;
            }

            hour = Node.Unspecified;
            end  = i;
            return false;
        }

        /// <summary>"half past", "quarter to", "ten past", "20 min past" — returns the offset and its direction.</summary>
        private readonly int TryRelativeMinutes(int i, out int minutes, out int direction, out int end)
        {
            minutes   = 0;
            direction = 0;
            end       = i;

            bool explicitUnit = false;
            int  at           = i;
            at = SkipWord(at, "a");

            if (AtTerm(at, TermKind.HalfWord))
            {
                minutes      = 30;
                explicitUnit = true;
                at++;
            }

            else if (AtTerm(at, TermKind.QuarterWord))
            {
                minutes      = 15;
                explicitUnit = true;
                at++;
            }
            else if (TryInteger(at, out int value, out int afterValue) && value > 0 && value < 60)
            {
                minutes = value;
                at      = afterValue;

                if (AtTermValue(at, TermKind.Unit, (int)TimeUnit.Minute)) { at++; explicitUnit = true; }
            }
            else
            {
                return -1;
            }

            if (AtTerm(at, TermKind.PastWord))      { direction =  1; at = After(at); }
            else if (AtTerm(at, TermKind.ToWord))   { direction = -1; at = After(at); }
            else                                    { return -1; }

            // "5 to 6pm" is a range, not five minutes to six: only a reading that cannot be an hour,
            // or one that says "minutes", counts backwards from the hour.
            if (direction < 0 && !explicitUnit && minutes <= 12) return -1;

            // "17 bis 18 Uhr" is a range too — a right-hand side past noon says both sides are hours
            if (direction < 0 && !explicitUnit && TryInteger(at, out int rightHour, out _) && rightHour > 12 && rightHour <= 24) return -1;

            end = at;
            return at;
        }

        /// <summary>True when something between the two indices marks the numbers as clock readings.</summary>
        private readonly bool HasClockMarker(int from, int to)
        {
            for (int k = from; k < to; k++)
            {
                if (AtTerm(k, TermKind.AmPm) && !AtTerm(k, TermKind.Filler)) return true;
                if (AtTerm(k, TermKind.OClock)) return true;
                if (AtTerm(k, TermKind.PartOfDay)) return true;
                if (At(k, LexKind.Colon)) return true;
                if (At(k, LexKind.Word) && (AtWord(k, "am") || AtWord(k, "pm") || AtWord(k, "a") || AtWord(k, "p"))) return true;
            }

            return false;
        }

        /// <summary>
        /// A four-digit number reads as a clock only inside an explicit range, and only when neither side of
        /// that range could be a year: "between 0730-0930" is a pair of clocks, "from 2015 and 2016" is not.
        /// </summary>
        private readonly bool IsInClockRangeContext(int i)
        {
            int other = -1;

            if ((At(i + 1, LexKind.Dash) || AtTerm(i + 1, TermKind.Connector)) && AtNumber(i + 2) && DigitsAt(i + 2) == 4)      other = i + 2;
            else if ((At(i - 1, LexKind.Dash) || AtTerm(i - 1, TermKind.Connector)) && AtNumber(i - 2) && DigitsAt(i - 2) == 4) other = i - 2;

            if (other < 0)
            {
                for (int k = Math.Max(0, i - 2); k < i; k++)
                {
                    if (AtTerm(k, TermKind.RangeStart)) return LooksLikeClock(i);
                }

                return false;
            }

            return LooksLikeClock(i) && LooksLikeClock(other);
        }

        /// <summary>A written year has no leading zero and sits in the calendar range; a clock reading does not.</summary>
        private readonly bool LooksLikeClock(int i)
        {
            if (_text[_lex[i].Start] == '0') return true;

            return NumberAt(i) < 1000 || NumberAt(i) > 3000;
        }

        // ------------------------------------------------------------------ a time

        private int TryTime(int i, out int node) => TryTime(i, out node, allowBareHour: false);

        /// <param name="allowBareHour">true inside an explicit range, where "from 9 to 11" really does mean nine o'clock.</param>
        private int TryTime(int i, out int node, bool allowBareHour)
        {
            node = Node.Unspecified;

            int at  = i;
            var mod = ModKind.None;

            bool approxIntroduced = false;

            if (AtTerm(at, TermKind.Approx))
            {
                mod = ModKind.Approx;
                at  = After(at);

                // "cerca de las tres" — the approximation keeps the introducer it was written with
                if (AtClockPrefix(at)) { at = After(at); approxIntroduced = true; }
            }

            var  pod       = PartOfDayKind.None;
            bool podLeading = false;

            int podEnd = TryPartOfDay(at, out var leadingPod, out _);
            if (podEnd > 0 && !IsMealTime(leadingPod))
            {
                // Only a lead-in when a clock reading follows: "in the morning at 7"
                int probe = podEnd;
                probe = SkipClockPrefix(SkipWords(probe, "at", "around"));

                int test = TryClock(probe, out _, out _, out _, out _, out _, out _, out _);
                if (test > 0)
                {
                    pod        = leadingPod;
                    podLeading = true;
                    at         = probe;
                }
            }

            int clockEnd = TryClock(at, out int hour, out int minute, out int second, out int ampm, out bool explicitMinutes, out bool marked, out bool dottedMinutes);
            if (clockEnd < 0) return -1;

            at = clockEnd;

            if (!podLeading)
            {
                // "2 nights" is a duration; only a marked clock or an introduced phrase takes a part of the day
                bool introduced = AtTerm(at, TermKind.Filler) || AtTerm(at, TermKind.Mod);

                // Inside a range each side is already known to be a clock, so "et 6 après-midi" qualifies it
                if (ampm >= 0 || explicitMinutes || marked || introduced || allowBareHour)
                {
                    int trailing = TryPartOfDay(at, out var trailingPod, out _);
                    if (trailing > 0)
                    {
                        pod = trailingPod;
                        at  = trailing;
                    }
                }
            }

            // A bare number is only a time when something marks it as one
            if (ampm < 0 && pod == PartOfDayKind.None && !explicitMinutes && !marked && !allowBareHour && !approxIntroduced && !ClockPrefixEndsAt(i)) return -1;

            var n = Node.Create(NodeKind.Time);

            // "at 6.45" reads as a clock because of the "at", so the "at" belongs to it
            n.LexStart  = dottedMinutes && AtWord(i - 1, "at") ? i - 1 : i;   // English writes "at 6.45", and keeps the "at"
            n.LexEnd    = at;
            n.Hour      = hour;
            n.Minute    = minute;
            n.Second    = second;
            n.AmPm      = ampm;
            n.PartOfDay = pod;
            n.Mod       = mod;
            SetSpan(ref n);
            node = Alloc(n);
            return at;
        }
    }
}
