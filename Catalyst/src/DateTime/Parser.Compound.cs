using System;

namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        // ------------------------------------------------------------------ time periods

        private int TryTimePeriod(int i, out int node) => TryTimePeriod(i, out node, allowBareHours: false);

        private int TryTimePeriod(int i, out int node, bool allowBareHours)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TryExplicitTimeRange(i, out int n1, allowBareHours), n1, ref best, ref bestNode);
            Consider(TryTimeWithDuration(i, out int n2),  n2, ref best, ref bestNode);
            Consider(TryPartOfDayPeriod(i, out int n4),   n4, ref best, ref bestNode);
            Consider(TryModTime(i, out int n3),           n3, ref best, ref bestNode);
            Consider(TryOpenEndedTime(i, out int n6),     n6, ref best, ref bestNode);

            if (best < 0)
            {
                node = Node.Unspecified;
                return -1;
            }

            if (AtTerm(best, TermKind.Mod, out int trailing))
            {
                var k = (ModKind)trailing;

                if ((k == ModKind.OrLater || k == ModKind.OrEarlier) && NodeAt(bestNode).Mod == ModKind.None && TryTime(After(best), out _) < 0 && TryRangeEndpoint(After(best), out _) < 0)
                {
                    ref var n = ref NodeAt(bestNode);
                    n.Mod    = k == ModKind.OrLater ? ModKind.Since : ModKind.Before;
                    n.LexEnd = After(best);
                    SetSpan(ref n);
                    best = n.LexEnd;
                }
            }

            node = bestNode;
            return best;
        }

        /// <summary>"3 pm or later" — a plain time that a trailing modifier opens into a period.</summary>
        private int TryOpenEndedTime(int i, out int node)
        {
            node = Node.Unspecified;

            int timeEnd = TryTime(i, out int time, allowBareHour: false);
            if (timeEnd < 0) return -1;

            if (!AtTerm(timeEnd, TermKind.Mod, out int trailing)) return -1;

            var k = (ModKind)trailing;
            if (k != ModKind.OrLater && k != ModKind.OrEarlier) return -1;

            var n = NodeAt(time);
            n.Kind     = NodeKind.TimeRange;
            n.LexStart = i;
            n.LexEnd   = After(timeEnd);
            n.Left     = time;
            n.Mod      = k == ModKind.OrLater ? ModKind.Since : ModKind.Before;
            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        /// <summary>"morning", "in the evening", "late afternoon", "at lunchtime", "in the night-time".</summary>
        private int TryPartOfDayPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            var mod = ModKind.None;
            int at  = i;

            if (AtTerm(at, TermKind.Approx))
            {
                mod = ModKind.Approx;
                at  = After(at);
            }

            int end = TryPartOfDay(at, out var kind, out var podMod, out int podStart);
            if (end < 0) return -1;
            if (podMod != ModKind.None && podStart > at) at = podStart;
            if (kind == PartOfDayKind.Noon || kind == PartOfDayKind.Midnight) return -1;

            // A word that primarily names a day ("mañana") is a date, not a part of the day
            for (int k = at; k < end; k++)
            {
                if (In(k) && _lex[k].Term.Kind == TermKind.SpecialDay) return -1;
            }

            if (podMod != ModKind.None) mod = podMod;

            int spanStart = podMod != ModKind.None && podStart > i ? podStart : i;

            // "early morning at 8:00" is one time, told apart from the evening by the part of the day
            if (AtWord(end, "at"))
            {
                int atTimeEnd = TryTime(end + 1, out int atTime, allowBareHour: true);

                if (atTimeEnd > 0)
                {
                    ref var t = ref NodeAt(atTime);
                    t.LexStart  = spanStart;
                    t.LexEnd    = atTimeEnd;
                    t.PartOfDay = kind;
                    SetSpan(ref t);
                    node = atTime;
                    return atTimeEnd;
                }
            }

            // "this evening from 7 to 9" — the part of the day says which of the two clock readings is meant
            int rangeEnd = TryExplicitTimeRange(SkipWord(end, "at"), out int range, allowBareHours: true);

            if (rangeEnd > 0)
            {
                ref var r = ref NodeAt(range);
                r.LexStart  = spanStart;
                r.LexEnd    = rangeEnd;
                r.PartOfDay = kind;
                SetSpan(ref r);
                node = range;
                return rangeEnd;
            }

            var n = Node.Create(NodeKind.TimeRange);
            n.LexStart  = spanStart;
            n.LexEnd    = end;
            n.PartOfDay = kind;
            n.Mod       = mod;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"5 to 6pm", "between 3:00 pm and 5:00 pm", "9-10 am", "from 2:30 to 2:15 pm".</summary>
        private int TryExplicitTimeRange(int i, out int node) => TryExplicitTimeRange(i, out node, allowBareHours: false);

        private int TryExplicitTimeRange(int i, out int node, bool allowBareHours)
        {
            node = Node.Unspecified;

            int at         = i;
            bool sawFrom   = false;
            bool sawBetween = false;

            if (AtWord(at, "any") && AtWord(at + 1, "time")) at += 2;

            if (AtTerm(at, TermKind.RangeStart, out int rangeKind))
            {
                sawFrom    = rangeKind == 0;
                sawBetween = rangeKind == 1;
                at         = After(at);
            }

            int leftEnd = TryTime(at, out int left, allowBareHour: true);
            if (leftEnd < 0) return -1;

            int mid = leftEnd;
            bool connector = false;

            if (sawBetween && AtWord(mid, "and"))                     { connector = true; mid++; }
            else if (AtTerm(mid, TermKind.Connector))                 { connector = true; mid = After(mid); }
            else if (At(mid, LexKind.Dash) || At(mid, LexKind.Tilde)) { connector = true; mid++; }

            if (!connector) return -1;

            int rightEnd = TryTime(mid, out int right, allowBareHour: true);
            if (rightEnd < 0) return -1;

            if (!sawFrom && !sawBetween && !allowBareHours && !HasClockMarker(i, rightEnd)) return -1;

            var n = Node.Create(NodeKind.TimeRange);
            n.LexStart = i;
            n.LexEnd   = rightEnd;
            n.Left     = left;
            n.Right    = right;
            SetSpan(ref n);
            node = Alloc(n);
            return rightEnd;
        }

        /// <summary>"for 2 hours from 2pm", "from 2pm for 2 hours", "from 9 for 2.5 hrs".</summary>
        private int TryTimeWithDuration(int i, out int node)
        {
            node = Node.Unspecified;

            int at       = i;
            int duration = Node.Unspecified;
            int time     = Node.Unspecified;
            int end;

            if (AtWord(at, "for"))
            {
                int durationEnd = TryDuration(at + 1, out duration);
                if (durationEnd < 0) return -1;

                int fromAt = durationEnd;
                if (!AtTerm(fromAt, TermKind.RangeStart)) return -1;
                fromAt = After(fromAt);

                int timeEnd = TryTime(fromAt, out time, allowBareHour: true);
                if (timeEnd < 0) return -1;

                end = timeEnd;
            }
            else if (AtTerm(at, TermKind.RangeStart, out int kind) && kind == 0)
            {
                int timeEnd = TryTime(After(at), out time, allowBareHour: true);
                if (timeEnd < 0) return -1;

                int forAt = timeEnd;
                if (!AtWord(forAt, "for")) return -1;

                int durationEnd = TryDuration(forAt + 1, out duration);
                if (durationEnd < 0) return -1;

                end = durationEnd;
            }
            else
            {
                return -1;
            }

            if (NodeAt(duration).Duration.IsDateOnly) return -1;

            var n = Node.Create(NodeKind.TimeRange);
            n.LexStart      = i;
            n.LexEnd        = end;
            n.Left          = time;
            n.RangeDuration = duration;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"after 3pm", "before 2.30pm", "as early as 7:00 am".</summary>
        private int TryModTime(int i, out int node)
        {
            node = Node.Unspecified;

            int at = i;

            if (AtWord(at, "any") && AtWord(at + 1, "time")) at += 2;

            var mod = ModKind.None;

            if (AtTerm(at, TermKind.Mod, out int modValue))
            {
                mod = (ModKind)modValue;
                at  = After(at);
            }
            else if (AtTerm(at, TermKind.RangeStart, out int kind) && kind == 0 && at > i)
            {
                mod = ModKind.Since;
                at  = After(at);
            }
            else
            {
                return -1;
            }

            if (mod == ModKind.OrLater || mod == ModKind.OrEarlier) return -1;

            bool approx = false;
            if (AtTerm(at, TermKind.Approx))
            {
                approx = true;
                at     = After(at);
            }

            bool narrowing = mod == ModKind.Early || mod == ModKind.Late || mod == ModKind.Mid;

            int time    = Node.Unspecified;
            int timeEnd = TryTime(at, out time);

            // A narrowing word in front of a part of the day keeps its own meaning
            if (narrowing && timeEnd < 0) time = Node.Unspecified;
            if (timeEnd < 0)
            {
                int podEnd = TryPartOfDayPeriod(at, out time);
                if (podEnd < 0) return -1;
                timeEnd = podEnd;
            }

            ref var inner = ref NodeAt(time);

            var n = inner;
            n.LexStart = i;
            n.LexEnd   = timeEnd;
            n.Kind     = NodeKind.TimeRange;
            n.Mod      = mod;
            n.Left     = inner.Kind == NodeKind.Time ? time : inner.Left;
            n.Right    = Node.Unspecified;

            if (inner.Kind == NodeKind.TimeRange && inner.PartOfDay != PartOfDayKind.None)
            {
                n.PartOfDay = inner.PartOfDay;
                n.Left      = Node.Unspecified;
            }

            if (approx) n.Mod = mod == ModKind.After ? ModKind.Approx : n.Mod;

            SetSpan(ref n);
            node = Alloc(n);
            return timeEnd;
        }

        // ------------------------------------------------------------------ date + time

        private int TryDateTime(int i, out int node)
        {
            node = Node.Unspecified;

            // "now", "right now", "at the moment", "end of day"
            if (AtTerm(i, TermKind.SpecialDay, out int special))
            {
                var kind = (SpecialDayKind)special;

                if (kind == SpecialDayKind.Now || kind == SpecialDayKind.EndOfDay)
                {
                    var present = Node.Create(NodeKind.DateTime);
                    present.LexStart = i;
                    present.LexEnd   = After(i);
                    present.Relative = kind == SpecialDayKind.Now ? RelativeKind.Current : RelativeKind.This;
                    present.PartOfDay = kind == SpecialDayKind.EndOfDay ? PartOfDayKind.Night : PartOfDayKind.None;
                    present.Hour     = kind == SpecialDayKind.EndOfDay ? 23 : Node.Unspecified;
                    present.Minute   = kind == SpecialDayKind.EndOfDay ? 59 : Node.Unspecified;
                    present.Second   = kind == SpecialDayKind.EndOfDay ? 59 : Node.Unspecified;
                    present.AmPm     = kind == SpecialDayKind.EndOfDay ? 1 : Node.Unspecified;
                    present.Holiday  = HolidayKind.None;
                    present.Day      = kind == SpecialDayKind.EndOfDay ? Node.Unspecified : Node.Unspecified;
                    present.Season   = SeasonKind.None;
                    present.OffsetDays = 0;
                    present.Anchor   = Node.Unspecified;
                    present.Mod      = ModKind.None;
                    SetSpan(ref present);
                    node = Alloc(present);
                    return present.LexEnd;
                }
            }

            // "last night at 8"
            int nightEnd = TryRelativeDayPartOfDay(i, out int nightNode);

            if (nightEnd > 0)
            {
                int at = SkipWord(nightEnd, "at");

                if (at != nightEnd)
                {
                    int timeEnd = TryTime(at, out int time, allowBareHour: true);

                    if (timeEnd > 0)
                    {
                        var n = NodeAt(nightNode);
                        ref var t = ref NodeAt(time);
                        n.Kind     = NodeKind.DateTime;
                        n.LexStart = i;
                        n.LexEnd   = timeEnd;
                        n.Hour     = t.Hour;
                        n.Minute   = t.Minute;
                        n.Second   = t.Second;
                        n.AmPm     = t.AmPm >= 0 ? t.AmPm : (n.PartOfDay == PartOfDayKind.Night || n.PartOfDay == PartOfDayKind.Tonight || n.PartOfDay == PartOfDayKind.Evening ? 1 : 0);
                        SetSpan(ref n);
                        node = Alloc(n);
                        return timeEnd;
                    }
                }
            }

            // "in 5 minutes", "30 min later", "half an hour from now", "3 minutes from now"
            int offsetEnd = TryTimeOffset(i, out int offsetNode);
            if (offsetEnd > 0)
            {
                node = offsetNode;
                return offsetEnd;
            }

            // "end of tomorrow", "the end of today", "end of this sunday"
            if (AtTerm(i, TermKind.Mod, out int endMod) && (ModKind)endMod == ModKind.End)
            {
                int inner = After(i);
                inner = SkipWords(inner, "of", "the");
                inner = SkipWord(inner, "the");

                int innerEnd = TryDate(inner, out int innerDate);
                if (innerEnd > 0)
                {
                    var eod = NodeAt(innerDate);
                    eod.Kind     = NodeKind.DateTime;
                    eod.LexStart = i;
                    eod.LexEnd   = innerEnd;
                    eod.Hour     = 23;
                    eod.Minute   = 59;
                    eod.Second   = 59;
                    eod.AmPm     = 1;
                    eod.Mod      = ModKind.None;
                    SetSpan(ref eod);
                    node = Alloc(eod);
                    return innerEnd;
                }
            }

            Span<int> dateEnds  = stackalloc int[8];
            Span<int> dateNodes = stackalloc int[8];

            int candidates = DateCandidates(i, dateEnds, dateNodes);
            int bestEnd    = -1;
            int bestNode   = Node.Unspecified;

            for (int c = 0; c < candidates; c++)
            {
                int dateEnd = dateEnds[c];
                int at      = dateEnd;
                bool marker = AtWord(at, "at") || At(at, LexKind.At);

                at = SkipWords(at, "at", "on");
                if (At(at, LexKind.Comma)) { at++; if (AtWord(at, "at")) marker = true; at = SkipWord(at, "at"); }
                if (At(at, LexKind.At))    { at++; marker = true; }
                if (AtWord(at, "at") || AtWord(at, "for") || AtTerm(at, TermKind.Approx)) marker = true;
                at = SkipWords(at, "at", "around");
                at = SkipWord(at, "for");

                int timeEnd = TryTime(at, out int time, allowBareHour: marker);

                // "sunday early morning at 8:00" — the part of the day only says which eight is meant
                if (timeEnd <= 0)
                {
                    int podEnd = TryPartOfDayPeriod(at, out int podTime);

                    if (podEnd > 0 && NodeAt(podTime).Kind == NodeKind.Time)
                    {
                        timeEnd = podEnd;
                        time    = podTime;
                    }
                }

                if (timeEnd <= 0) continue;

                var n = NodeAt(dateNodes[c]);
                ref var t = ref NodeAt(time);
                n.Kind      = NodeKind.DateTime;
                n.LexStart  = i;
                n.LexEnd    = timeEnd;
                n.Hour      = t.Hour;
                n.Minute    = t.Minute;
                n.Second    = t.Second;
                n.AmPm      = t.AmPm;
                n.PartOfDay = t.PartOfDay;

                // "wed oct 26 15:50:06 2016" — a trailing year after the clock
                int yearAt = timeEnd;
                if (n.Year < 0 && TryYear(yearAt, out int trailingYear, out int yearEnd))
                {
                    n.Year   = trailingYear;
                    n.LexEnd = yearEnd;
                }

                if (n.LexEnd <= bestEnd) continue;

                SetSpan(ref n);
                bestEnd  = n.LexEnd;
                bestNode = Alloc(n);
            }

            if (bestEnd > 0)
            {
                node = bestNode;
                return bestEnd;
            }

            // "3pm today", "8am this morning", "10, tonight", "tomorrow, 13/04/21, at 7 pm"
            int leadingTimeEnd = TryTime(i, out int leadingTime);
            bool leadingWasBare = false;

            if (leadingTimeEnd < 0)
            {
                leadingTimeEnd  = TryTime(i, out leadingTime, allowBareHour: true);
                leadingWasBare  = leadingTimeEnd > 0;
            }

            if (leadingTimeEnd > 0)
            {
                int at = leadingTimeEnd;
                if (At(at, LexKind.Comma)) at++;
                at = SkipWords(at, "on", "of");

                int trailingDate    = Node.Unspecified;
                int trailingDateEnd = leadingWasBare ? -1 : TryDate(at, out trailingDate);

                if (trailingDateEnd < 0)
                {
                    trailingDateEnd = TryRelativeDayPartOfDay(at, out trailingDate);
                    if (trailingDateEnd < 0) trailingDateEnd = TryBarePartOfDayAsToday(at, out trailingDate);
                }

                if (trailingDateEnd > 0)
                {
                    var n = NodeAt(trailingDate);
                    ref var t = ref NodeAt(leadingTime);
                    n.Kind      = NodeKind.DateTime;
                    n.LexStart  = i;
                    n.LexEnd    = trailingDateEnd;
                    n.Hour      = t.Hour;
                    n.Minute    = t.Minute;
                    n.Second    = t.Second;
                    n.AmPm      = t.AmPm;
                    if (t.PartOfDay != PartOfDayKind.None) n.PartOfDay = t.PartOfDay;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return trailingDateEnd;
                }
            }

            return -1;
        }

        /// <summary>"in 5 minutes", "30 min later", "half an hour from now", "within 3 min".</summary>
        private int TryTimeOffset(int i, out int node)
        {
            node = Node.Unspecified;

            int at    = i;
            bool sawIn = false;

            if (AtWord(at, "in") || AtWord(at, "within"))
            {
                sawIn = true;
                at++;
            }

            int durationEnd = TryDuration(at, out int duration);
            if (durationEnd < 0) return -1;

            ref var d = ref NodeAt(duration);
            if (d.Duration.IsDateOnly) return -1;

            int  sign = 0;
            int  end  = durationEnd;

            if (AtTerm(durationEnd, TermKind.Ago))          { sign = -1; end = After(durationEnd); }
            else if (AtTerm(durationEnd, TermKind.FromNow)) { sign =  1; end = After(durationEnd); }
            else if (AtWord(durationEnd, "from") && AtTermValue(durationEnd + 1, TermKind.SpecialDay, (int)SpecialDayKind.Now))
            {
                sign = 1;
                end  = After(durationEnd + 1);
            }
            else if (sawIn) { sign = 1; }
            else            { return -1; }

            var n = Node.Create(NodeKind.DateTime);
            n.LexStart        = i;
            n.LexEnd          = end;
            n.Relative        = RelativeKind.Current;
            n.Left            = duration;
            n.OffsetDays      = sign;    // carries the direction; the resolver reads Left for the amount
            n.Mod             = ModKind.None;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        // ------------------------------------------------------------------ date + time period

        /// <summary>"from 2pm till tomorrow 4:30pm", "between 2:00 tomorrow and 4:00", "between now and eight o'clock".</summary>
        private int TryExplicitMomentRange(int i, out int node)
        {
            node = Node.Unspecified;

            int at         = i;
            bool sawFrom   = false;
            bool sawBetween = false;

            if (AtTerm(at, TermKind.RangeStart, out int rangeKind))
            {
                sawFrom    = rangeKind == 0;
                sawBetween = rangeKind == 1;
                at         = After(at);
            }

            if (!sawFrom && !sawBetween) return -1;

            int leftEnd = TryMoment(at, out int left);
            if (leftEnd < 0) return -1;

            int mid = leftEnd;
            bool connector = false;

            if (sawBetween && AtWord(mid, "and"))                            { connector = true; mid++; }
            else if (AtTerm(mid, TermKind.Connector) && !AtWord(mid, "and")) { connector = true; mid = After(mid); }
            else if (At(mid, LexKind.Dash))                                  { connector = true; mid++; }

            if (!connector) return -1;

            int rightEnd = TryMoment(mid, out int right);
            if (rightEnd < 0) return -1;

            ref var l = ref NodeAt(left);
            ref var r = ref NodeAt(right);

            // Only when one end is a complete moment; two clock readings alone are a time range,
            // and two days alone are a date range.
            if (l.Kind != NodeKind.DateTime && r.Kind != NodeKind.DateTime) return -1;
            if (!l.HasAnyTime && !r.HasAnyTime) return -1;

            var n = Node.Create(NodeKind.DateTimeRange);
            n.LexStart           = i;
            n.LexEnd             = rightEnd;
            n.Left               = left;
            n.Right              = right;
            n.ChildrenAreMoments = true;
            SetSpan(ref n);
            node = Alloc(n);
            return rightEnd;
        }

        /// <summary>One end of a moment range: a datetime, a clock reading, or a day.</summary>
        private int TryMoment(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TryDateTime(i, out int n1), n1, ref best, ref bestNode);
            Consider(TryTime(i, out int n2, allowBareHour: true), n2, ref best, ref bestNode);
            Consider(TryDate(i, out int n3),     n3, ref best, ref bestNode);

            node = bestNode;
            return best;
        }

        private int TryDateTimePeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int momentRange = TryExplicitMomentRange(i, out int momentNode);
            if (momentRange > 0)
            {
                node = momentNode;
                return momentRange;
            }

            // "within 2h", "next hour", "last minute", "5 coming minutes"
            int clockPeriodEnd = TryClockRelativePeriod(i, out int clockPeriod);
            if (clockPeriodEnd > 0)
            {
                node = clockPeriod;
                return clockPeriodEnd;
            }

            Span<int> dateEnds  = stackalloc int[8];
            Span<int> dateNodes = stackalloc int[8];

            int candidates = DateCandidates(i, dateEnds, dateNodes);
            int bestEnd    = -1;
            int bestNode   = Node.Unspecified;

            for (int c = 0; c < candidates; c++)
            {
                int at = dateEnds[c];
                if (At(at, LexKind.Comma)) at++;
                at = SkipWord(at, "at");

                int periodEnd = TryTimePeriod(at, out int period, allowBareHours: true);
                if (periodEnd <= bestEnd) continue;
                if (NodeAt(period).Kind != NodeKind.TimeRange) continue;   // a single time makes a datetime, not a range

                var n = NodeAt(dateNodes[c]);
                ref var p = ref NodeAt(period);
                n.Kind      = NodeKind.DateTimeRange;
                n.LexStart  = i;
                n.LexEnd    = periodEnd;
                n.PartOfDay     = p.PartOfDay;
                n.Left          = p.Left;
                n.Right         = p.Right;
                n.RangeDuration = p.RangeDuration;
                n.Mod           = p.Mod;
                SetSpan(ref n);
                bestEnd  = periodEnd;
                bestNode = Alloc(n);
            }

            if (bestEnd > 0)
            {
                node = bestNode;
                return bestEnd;
            }

            int leadPeriodEnd = TryTimePeriod(i, out int leadPeriod, allowBareHours: true);

            if (leadPeriodEnd > 0)
            {
                ref var p = ref NodeAt(leadPeriod);

                // A part of the day on its own resolves against today
                if (p.PartOfDay == PartOfDayKind.Tonight)
                {
                    var tonight = p;
                    tonight.Kind     = NodeKind.DateTimeRange;
                    tonight.Relative = RelativeKind.Current;
                    tonight.LexStart = i;
                    tonight.LexEnd   = leadPeriodEnd;
                    SetSpan(ref tonight);
                    node = Alloc(tonight);
                    return leadPeriodEnd;
                }

                if (IsMealTime(p.PartOfDay)) return -1;

                int at = leadPeriodEnd;
                if (At(at, LexKind.Comma)) at++;
                at = SkipWords(at, "on", "of");

                int trailingDateEnd = TryDate(at, out int trailingDate);

                if (trailingDateEnd > 0)
                {
                    // "from 3-8pm yesterday afternoon" — the part of the day only repeats what the clock said
                    var tailKind    = PartOfDayKind.None;
                    int trailingPod = p.PartOfDay == PartOfDayKind.None
                                    ? TryPartOfDay(SkipWords(trailingDateEnd, "in", "the"), out tailKind, out _)
                                    : -1;

                    if (trailingPod > 0 && !IsMealTime(tailKind)) trailingDateEnd = trailingPod;

                    var n = NodeAt(trailingDate);
                    n.Kind      = NodeKind.DateTimeRange;
                    n.LexStart  = i;
                    n.LexEnd    = trailingDateEnd;
                    n.PartOfDay     = p.PartOfDay;
                    n.Left          = p.Left;
                    n.Right         = p.Right;
                    n.RangeDuration = p.RangeDuration;
                    n.Mod           = p.Mod;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return trailingDateEnd;
                }
            }

            // "today pm", "tomorrow am" — never "9-10 am" or "mon 9 am", where the number is the clock
            for (int c = 0; c < candidates; c++)
            {
                int at = dateEnds[c];

                ref var candidate = ref NodeAt(dateNodes[c]);
                if (candidate.Day >= 0 || candidate.Month >= 0 || candidate.Year >= 0) continue;

                if (TryAmPm(at, out int half, out int halfEnd) && _lex[at].SpaceBefore)
                {
                    var n = NodeAt(dateNodes[c]);
                    n.Kind      = NodeKind.DateTimeRange;
                    n.LexStart  = i;
                    n.LexEnd    = halfEnd;
                    n.PartOfDay = half == 0 ? PartOfDayKind.Morning : PartOfDayKind.Afternoon;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return halfEnd;
                }
            }

            // "this afternoon", "tomorrow night" — a relative day with a part of the day
            int relDateEnd = TryRelativeDayPartOfDay(i, out int relDate);
            if (relDateEnd > 0)
            {
                node = relDate;
                return relDateEnd;
            }

            return -1;
        }

        /// <summary>A part of the day standing on its own resolves against today ("10, tonight").</summary>
        private int TryBarePartOfDayAsToday(int i, out int node)
        {
            node = Node.Unspecified;

            int end = TryPartOfDay(i, out var kind, out var mod);
            if (end < 0) return -1;
            if (kind != PartOfDayKind.Tonight) return -1;

            var n = Node.Create(NodeKind.Date);
            n.LexStart   = i;
            n.LexEnd     = end;
            n.Relative   = RelativeKind.Current;
            n.OffsetDays = 0;
            n.PartOfDay  = kind;
            n.Mod        = mod;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>Every date reading at <paramref name="i"/>, so a following clock can pick the one that fits.</summary>
        private int DateCandidates(int i, scoped Span<int> ends, scoped Span<int> nodes)
        {
            int count = 0;

            int dateEnd = TryDate(i, out int n1);
            if (dateEnd > 0 && NodeAt(n1).Holiday != HolidayKind.None) dateEnd = -1;
            AddCandidate(dateEnd, n1, ends, nodes, ref count);
            AddCandidate(TryWeekdayDate(i, out int n2),   n2, ends, nodes, ref count);
            AddCandidate(TryWeekdayBare(i, out int n3),   n3, ends, nodes, ref count);
            AddCandidate(TrySpecialDay(i, out int n4),    n4, ends, nodes, ref count);
            AddCandidate(TryMonthNameDate(i, out int n5), n5, ends, nodes, ref count);
            AddCandidate(TryNumericDate(i, out int n6),   n6, ends, nodes, ref count);
            AddCandidate(TryBarePartOfDayAsToday(i, out int n8), n8, ends, nodes, ref count);

            return count;
        }

        private static void AddCandidate(int end, int node, scoped Span<int> ends, scoped Span<int> nodes, ref int count)
        {
            if (end <= 0 || node < 0 || count >= ends.Length) return;

            for (int k = 0; k < count; k++)
            {
                if (ends[k] == end) return;
            }

            ends[count]  = end;
            nodes[count] = node;
            count++;
        }

        private static bool IsMealTime(PartOfDayKind kind) =>
            kind is PartOfDayKind.Lunch or PartOfDayKind.Dinner or PartOfDayKind.Breakfast or PartOfDayKind.Brunch;

        private int TryRelativeDayPartOfDay(int i, out int node)
        {
            node = Node.Unspecified;

            if (!AtTerm(i, TermKind.Relative, out int relValue)) return -1;

            int at = After(i);

            int podEnd = TryPartOfDay(at, out var kind, out var mod);
            if (podEnd < 0) return -1;

            var rel = (RelativeKind)relValue;

            var n = Node.Create(NodeKind.DateTimeRange);
            n.LexStart   = i;
            n.LexEnd     = podEnd;
            n.PartOfDay  = kind;
            n.Relative   = RelativeKind.Current;
            n.OffsetDays = rel switch
            {
                RelativeKind.Next or RelativeKind.Coming or RelativeKind.Following =>  1,
                RelativeKind.Last or RelativeKind.Previous                         => -1,
                _                                                                  =>  0,
            };
            n.Mod = mod;

            // "next evening from 7 to 9" — the part of the day says which of the two clock readings is meant
            int rangeEnd = TryExplicitTimeRange(SkipWord(podEnd, "at"), out int range, allowBareHours: true);

            if (rangeEnd > 0)
            {
                ref var r = ref NodeAt(range);
                n.Left          = r.Left;
                n.Right         = r.Right;
                n.RangeDuration = r.RangeDuration;
                n.LexEnd        = rangeEnd;
            }

            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        /// <summary>"next hour", "last minute", "within 2h", "5 coming minutes", "13 last minutes".</summary>
        private int TryClockRelativePeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at    = SkipArticle(i);
            if (at != i && !_lexicon.ArticleInPeriodSpan) i = at;
            int count = Node.Unspecified;
            var rel   = RelativeKind.None;
            bool within = false;

            if (AtWord(at, "within"))
            {
                within = true;
                at++;
                at = SkipArticle(at);

                if (AtTerm(at, TermKind.Relative, out int wRel))
                {
                    rel = (RelativeKind)wRel;
                    at  = After(at);
                }
            }

            bool countLedTheUnit = false;

            if (count < 0 && TryInteger(at, out int leading, out int afterLeading) && leading > 0 && leading < 1000 && (AtTerm(afterLeading, TermKind.Relative) || within))
            {
                count           = leading;
                at              = afterLeading;
                countLedTheUnit = true;
            }

            if (AtTerm(at, TermKind.Relative, out int relValue))
            {
                rel = (RelativeKind)relValue;
                at  = After(at);
            }

            if (count < 0 && TryInteger(at, out int trailing, out int afterTrailing) && trailing > 0 && trailing < 1000)
            {
                count = trailing;
                at    = afterTrailing;
            }

            if (!AtTerm(at, TermKind.Unit, out int unitValue)) return -1;

            var unit = (TimeUnit)unitValue;
            if (unit != TimeUnit.Hour && unit != TimeUnit.Minute && unit != TimeUnit.Second) return -1;

            // "13 last minute" is the number thirteen beside "last minute", not thirteen minutes
            if (countLedTheUnit && count > 1 && !within && !LooksPlural(at)) return -1;
            if (rel == RelativeKind.None && !within) return -1;

            // "last two hours" is the verb "last"; a period written that way says "the" or uses digits
            if (count >= 0 && (rel == RelativeKind.Last || rel == RelativeKind.Previous || rel == RelativeKind.JustPast) && !AtWord(i, "the") && !AtNumber(i)) return -1;

            at++;

            var n = Node.Create(NodeKind.DateTimeRange);
            n.LexStart    = i;
            n.LexEnd      = at;
            n.PeriodUnit  = unit;
            n.PeriodCount = count < 0 ? 1 : count;
            n.Relative    = within ? RelativeKind.Next : (rel == RelativeKind.None ? RelativeKind.Next : rel);
            SetSpan(ref n);
            node = Alloc(n);
            return at;
        }

        // ------------------------------------------------------------------ recurring sets

        private int TrySet(int i, out int node)
        {
            node = Node.Unspecified;

            int at       = i;
            int interval = 1;
            bool marked  = false;

            // "19th of every month"
            int ordinalDay = Node.Unspecified;
            if (TryOrdinal(at, out int ord, out int afterOrd) && ord >= 1 && ord <= 31 && (AtWord(afterOrd, "of") || AtTerm(afterOrd, TermKind.SetPrefix)))
            {
                int probe = SkipWord(afterOrd, "of");

                if (AtTerm(probe, TermKind.SetPrefix))
                {
                    ordinalDay = ord;
                    at         = probe;
                }
            }

            int prefixKind = -1;

            if (AtTerm(at, TermKind.SetPrefix, out prefixKind))
            {
                marked = true;
                at     = After(at);
                at     = SkipGlue(at, 1);

                if (AtWord(at, "other"))
                {
                    interval = 2;
                    at++;
                }
                else if (TryInteger(at, out int every, out int afterEvery) && every > 0 && every < 100)
                {
                    interval = every;
                    at       = afterEvery;
                }
            }
            else if (AtWord(at, "once"))
            {
                marked = true;
                at++;
                at = SkipWords(at, "a", "an");
                at = SkipWords(at, "per", "every");
            }

            // "weekly", "daily", "annually", "quarterly"
            if (AtTerm(at, TermKind.SetFrequency, out int frequency))
            {
                var freq = Node.Create(NodeKind.Set);
                freq.LexStart    = i;
                freq.LexEnd      = After(at);
                freq.SetUnit     = (TimeUnit)frequency;
                freq.SetInterval = interval;
                SetSpan(ref freq);
                node = Alloc(freq);
                return freq.LexEnd;
            }

            bool bareWeekends = !marked && AtTermValue(at, TermKind.Unit, (int)TimeUnit.Weekend)
                                && _text.Slice(_lex[at].Start, _lex[at].Length).EndsWith("s", StringComparison.OrdinalIgnoreCase);

            if (bareWeekends)
            {
                var weekends = Node.Create(NodeKind.Set);
                weekends.LexStart    = i;
                weekends.LexEnd      = at + 1;
                weekends.SetUnit     = TimeUnit.Weekend;
                weekends.SetInterval = 1;
                SetSpan(ref weekends);
                node = Alloc(weekends);
                return weekends.LexEnd;
            }

            if (!marked && !AtTerm(at, TermKind.Weekday) && !AtTerm(at, TermKind.PartOfDay)) return -1;

            var n = Node.Create(NodeKind.Set);
            n.LexStart    = i;
            n.SetInterval = interval;
            n.Day         = ordinalDay;

            int end;

            if (AtTerm(at, TermKind.Weekday, out int weekday))
            {
                bool plural     = _text.Slice(_lex[at].Start, _lex[at].Length).EndsWith("s", StringComparison.OrdinalIgnoreCase);
                bool pluralPart = AtTerm(at + 1, TermKind.PartOfDay) && _text.Slice(_lex[at + 1].Start, _lex[at + 1].Length).EndsWith("s", StringComparison.OrdinalIgnoreCase);

                if (!marked && !plural && !pluralPart) return -1;

                n.Weekday = weekday;
                n.SetUnit = TimeUnit.Week;
                end       = at + 1;
            }
            else if (AtTerm(at, TermKind.Unit, out int unitValue))
            {
                if (!marked) return -1;

                var unit = (TimeUnit)unitValue;
                bool plural = _text.Slice(_lex[at].Start, _lex[at].Length).EndsWith("s", StringComparison.OrdinalIgnoreCase);

                // "all day" / "all month" are durations; only "every"/"each" turn a bare unit into a recurrence
                if (prefixKind != 0 && unit != TimeUnit.Weekend && unit != TimeUnit.Year && !plural) return -1;

                n.SetUnit = unit;
                end       = at + 1;
            }
            else if (AtTerm(at, TermKind.PartOfDay, out int podValue))
            {
                bool plural = _text.Slice(_lex[at].Start, _lex[at].Length).EndsWith("s", StringComparison.OrdinalIgnoreCase);
                if (!marked && !plural) return -1;

                n.PartOfDay = (PartOfDayKind)podValue;
                n.SetUnit   = TimeUnit.Day;
                end         = After(at);
            }
            else if (AtTerm(at, TermKind.BusinessDay))
            {
                n.SetUnit      = TimeUnit.Day;
                n.BusinessDays = true;
                end            = at + 1;
            }
            else
            {
                return -1;
            }

            // "every monday at 4pm", "tuesdays at 9am", "friday mornings"
            int tail = end;
            tail = SkipWords(tail, "at", "on");

            int timeEnd = TryTime(tail, out int time);
            if (timeEnd > 0)
            {
                ref var t = ref NodeAt(time);
                n.Hour   = t.Hour;
                n.Minute = t.Minute;
                n.Second = t.Second;
                n.AmPm   = t.AmPm;
                end      = timeEnd;
            }
            else
            {
                int podEnd = TryPartOfDayPeriod(tail, out int pod);
                if (podEnd > 0 && n.PartOfDay == PartOfDayKind.None)
                {
                    n.PartOfDay = NodeAt(pod).PartOfDay;
                    end         = podEnd;
                }
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"3pm each day", "6am everyday", "every day at 7:13 p.m." — a time that carries a recurrence.</summary>
        private int TrySetWithLeadingTime(int i, out int node)
        {
            node = Node.Unspecified;

            int timeEnd = TryTime(i, out int time);
            if (timeEnd < 0) return -1;

            int setEnd = TrySet(timeEnd, out int set);
            if (setEnd < 0) return -1;

            ref var t = ref NodeAt(time);

            var n = NodeAt(set);
            n.LexStart = i;
            n.LexEnd   = setEnd;
            n.Hour     = t.Hour;
            n.Minute   = t.Minute;
            n.Second   = t.Second;
            n.AmPm     = t.AmPm;
            SetSpan(ref n);
            node = Alloc(n);
            return setEnd;
        }
    }
}
