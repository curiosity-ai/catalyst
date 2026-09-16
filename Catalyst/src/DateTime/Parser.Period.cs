using System;

namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        /// <summary>Matches any date period ("last week", "april 2017", "from 2014 to 2018"), keeping the longest.</summary>
        private int TryDatePeriod(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TryDayRangeInMonth(i, out int n10),  n10, ref best, ref bestNode);
            Consider(TryExplicitDateRange(i, out int n1),  n1, ref best, ref bestNode);
            Consider(TryModDatePeriod(i, out int n2),      n2, ref best, ref bestNode);
            Consider(TryNthPeriodOf(i, out int n3),        n3, ref best, ref bestNode);
            Consider(TryWeekOfDate(i, out int n14),       n14, ref best, ref bestNode);
            Consider(TryDurationFromDate(i, out int n4),   n4, ref best, ref bestNode);
            Consider(TryShorthandPeriod(i, out int n11),  n11, ref best, ref bestNode);
            Consider(TryComparisonPeriod(i, out int n13), n13, ref best, ref bestNode);
            Consider(TryTrailingModDate(i, out int n15),  n15, ref best, ref bestNode);
            Consider(TryHolidayWeekend(i, out int n12),  n12, ref best, ref bestNode);
            Consider(TrySimplePeriod(i, out int n5),       n5, ref best, ref bestNode);

            if (best < 0)
            {
                node = Node.Unspecified;
                return -1;
            }

            // "2018 or later", "1/1/2016 and after"
            if (AtTerm(best, TermKind.Mod, out int trailing))
            {
                var k = (ModKind)trailing;

                // "2018 or later" closes the period; "before 2010 or after 2018" is two periods
                if ((k == ModKind.OrLater || k == ModKind.OrEarlier) && NodeAt(bestNode).Mod == ModKind.None && TryRangeEndpoint(After(best), out _) < 0)
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

        /// <summary>"&lt;=2019", "=2019", "&gt; = 2019" — a year written as a comparison.</summary>
        private int TryComparisonPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = i;
            var mod = ModKind.None;

            if (At(at, LexKind.Less))         { mod = ModKind.Before; at++; }
            else if (At(at, LexKind.Greater)) { mod = ModKind.After;  at++; }
            else if (!At(at, LexKind.Equal))  { return -1; }

            if (At(at, LexKind.Equal))
            {
                // "≥ 2019" takes in 2019 itself, where "> 2019" starts after it
                if (mod == ModKind.After)       mod = ModKind.Since;
                else if (mod == ModKind.Before) mod = ModKind.Until;
                at++;
            }
            else if (mod == ModKind.None) { return -1; }

            if (at == i) return -1;

            int inner = TryRangeEndpoint(at, out int child);
            if (inner < 0) return -1;

            var n = NodeAt(child);
            n.Kind     = NodeKind.DateRange;
            n.LexStart = i;
            n.LexEnd   = inner;
            n.Mod      = mod;
            SetSpan(ref n);
            node = Alloc(n);
            return inner;
        }

        /// <summary>"the week of april 10th", "the week of the 18th", "w/c feb 4", "the week beginning february 4".</summary>
        private int TryWeekOfDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);

            if (AtWord(at, "w") && At(at + 1, LexKind.Slash) && AtWord(at + 2, "c"))
            {
                at += 3;
            }
            else
            {
                if (!AtTermValue(at, TermKind.Unit, (int)TimeUnit.Week)) return -1;

                at++;

                if (AtWord(at, "of"))                                                                  { at++; }
                else if (AtWord(at, "beginning") || AtWord(at, "commencing") || AtWord(at, "starting")) { at++; at = SkipWord(at, "on"); }
                else                                                                                   { return -1; }
            }

            int dateEnd = TryDate(at, out int date);
            if (dateEnd < 0) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart     = i;
            n.LexEnd       = dateEnd;
            n.Anchor       = date;
            n.PeriodUnit   = TimeUnit.Week;
            n.PeriodCount  = 1;
            SetSpan(ref n);
            node = Alloc(n);
            return dateEnd;
        }

        /// <summary>"1/1/2016 and after" — a plain date that a trailing modifier opens into a period.</summary>
        private int TryTrailingModDate(int i, out int node)
        {
            node = Node.Unspecified;

            int dateEnd = TryDate(i, out int date);
            if (dateEnd < 0) return -1;

            if (!AtTerm(dateEnd, TermKind.Mod, out int trailing)) return -1;

            var k = (ModKind)trailing;
            if (k != ModKind.OrLater && k != ModKind.OrEarlier) return -1;

            var n = NodeAt(date);
            n.Kind     = NodeKind.DateRange;
            n.LexStart = i;
            n.LexEnd   = After(dateEnd);
            n.Mod      = k == ModKind.OrLater ? ModKind.Since : ModKind.Before;
            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        /// <summary>"eoy", "end of year", "to date", "year to date" — periods written as a fixed phrase.</summary>
        private int TryShorthandPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            if (!_lexicon.ArticleInPeriodSpan) i = SkipArticle(i);   // "the year to date" is "year to date"

            var  unit = TimeUnit.None;
            var  mod  = ModKind.None;
            int  end  = i;

            if (AtWord(i, "eoy"))      { unit = TimeUnit.Year;  mod = ModKind.End; end = i + 1; }
            else if (AtWord(i, "eom")) { unit = TimeUnit.Month; mod = ModKind.End; end = i + 1; }
            else if (AtWord(i, "eow")) { unit = TimeUnit.Week;  mod = ModKind.End; end = i + 1; }
            else if (AtWord(i, "ytd")) { unit = TimeUnit.Year;  mod = ModKind.Until; end = i + 1; }
            else if (AtTerm(i, TermKind.Mod, out int modValue) && (ModKind)modValue == ModKind.End)
            {
                int at = After(i);
                at = SkipWords(at, "of", "the");
                at = SkipArticle(at);

                if (AtTerm(at, TermKind.Unit, out int unitValue) && !AtTerm(at, TermKind.Relative))
                {
                    unit = (TimeUnit)unitValue;
                    mod  = ModKind.End;
                    end  = at + 1;
                }
            }
            else if ((AtWord(i, "to") || AtWord(i, "till") || AtWord(i, "until")) && AtWord(i + 1, "date"))
            {
                unit = TimeUnit.Year;
                mod  = ModKind.Until;
                end  = i + 2;
            }
            else if (AtTermValue(i, TermKind.Unit, (int)TimeUnit.Year) && AtWord(i + 1, "to") && AtWord(i + 2, "date"))
            {
                unit = TimeUnit.Year;
                mod  = ModKind.Until;
                end  = i + 3;
            }

            if (unit == TimeUnit.None) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart    = i;
            n.LexEnd      = end;
            n.PeriodUnit  = unit;
            n.PeriodCount = 1;
            n.Relative    = RelativeKind.This;
            n.Mod         = mod;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"labor day weekend", "the weekend of halloween", "halloween weekend 2021".</summary>
        private int TryHolidayWeekend(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);
            int holidayEnd;
            int holiday;

            if (AtTermValue(at, TermKind.Unit, (int)TimeUnit.Weekend))
            {
                int afterWeekend = SkipWord(at + 1, "of");
                holidayEnd = TryHolidayDate(afterWeekend, out holiday);
                if (holidayEnd < 0) return -1;
            }
            else
            {
                holidayEnd = TryHolidayDate(at, out holiday);
                if (holidayEnd < 0) return -1;
                if (!AtTermValue(holidayEnd, TermKind.Unit, (int)TimeUnit.Weekend)) return -1;
                holidayEnd++;
            }

            if (TryYear(holidayEnd, out int weekendYear, out int weekendYearEnd))
            {
                ref var h = ref NodeAt(holiday);
                h.Year    = weekendYear;
                holidayEnd = weekendYearEnd;
            }

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart   = i;
            n.LexEnd     = holidayEnd;
            n.Left       = holiday;
            n.PeriodUnit = TimeUnit.Weekend;
            SetSpan(ref n);
            node = Alloc(n);
            return holidayEnd;
        }

        /// <summary>
        /// Two days of the same month: "between 3 and 12 of sept", "april 9th through 17th", "november 19-20".
        /// The month is written once, so neither endpoint parses as a date on its own.
        /// </summary>
        private int TryDayRangeInMonth(int i, out int node)
        {
            node = Node.Unspecified;

            int  at         = i;
            bool sawBetween = false;

            if (AtTerm(at, TermKind.RangeStart, out int rangeKind))
            {
                sawBetween = rangeKind == 1;
                at         = After(at);
            }

            at = SkipWord(at, "on");
            at = SkipArticle(at);

            int month = Node.Unspecified;
            int leadingMonthAt = Node.Unspecified;

            if (AtTerm(at, TermKind.Month, out int leadMonth))
            {
                month          = leadMonth;
                leadingMonthAt = at;
                at++;
                at = SkipArticle(at);
            }

            if (!TryDayNumber(at, out int firstDay, out int afterFirst)) return -1;

            int mid = afterFirst;
            bool connector = false;

            if (sawBetween && AtWord(mid, "and"))                            { connector = true; mid++; }
            else if (AtTerm(mid, TermKind.Connector) && !AtWord(mid, "and")) { connector = true; mid = After(mid); }
            else if (At(mid, LexKind.Dash) || At(mid, LexKind.Tilde))        { connector = true; mid++; }

            if (!connector) return -1;

            mid = SkipArticle(mid);

            if (!TryDayNumber(mid, out int secondDay, out int afterSecond)) return -1;

            int end = afterSecond;

            if (month < 0)
            {
                int ofAt = SkipWords(end, "of", "in");
                ofAt = SkipArticle(ofAt);

                if (!AtTerm(ofAt, TermKind.Month, out month)) return -1;

                end = ofAt + 1;
            }

            int year = Node.Unspecified;
            int yearAt = end;
            if (At(yearAt, LexKind.Comma)) yearAt++;
            yearAt = SkipWords(yearAt, "of", "in");

            if (TryYearLoose(yearAt, out int parsedYear, out int yearEnd))
            {
                year = parsedYear;
                end  = yearEnd;
            }

            if (leadingMonthAt >= 0 && secondDay < firstDay) return -1;

            var left = Node.Create(NodeKind.Date);
            left.LexStart = i;
            left.LexEnd   = afterFirst;
            left.Month    = month;
            left.Day      = firstDay;
            left.Year     = year;
            SetSpan(ref left);

            var right = Node.Create(NodeKind.Date);
            right.LexStart = mid;
            right.LexEnd   = afterSecond;
            right.Month    = month;
            right.Day      = secondDay;
            right.Year     = year;
            SetSpan(ref right);

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.LexEnd   = end;
            n.Left     = Alloc(left);
            n.Right    = Alloc(right);
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>One endpoint of an explicit range: a date, or a period that is not itself a range.</summary>
        private int TryRangeEndpoint(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            if (_modDepth == 0)
            {
                _modDepth++;
                Consider(TryModDatePeriod(i, out int nm), nm, ref best, ref bestNode);
                Consider(TryShorthandPeriod(i, out int ns), ns, ref best, ref bestNode);
                _modDepth--;
            }

            Consider(TryNthPeriodOf(i, out int n1),  n1, ref best, ref bestNode);
            Consider(TrySimplePeriod(i, out int n2), n2, ref best, ref bestNode);
            Consider(TryDate(i, out int n3),         n3, ref best, ref bestNode);
            Consider(TryNowAsDate(i, out int n4),    n4, ref best, ref bestNode);

            node = bestNode;
            return best;
        }

        /// <summary>"now" / "today" used as the open end of a range: "between jan 22 and now".</summary>
        private int TryNowAsDate(int i, out int node)
        {
            node = Node.Unspecified;

            bool current = AtTermValue(i, TermKind.SpecialDay, (int)SpecialDayKind.Now);

            if (!current && !(AtWord(i, "current") && AtWord(i + 1, "date")) && !AtWord(i, "date")) return -1;

            int end = current ? After(i) : (AtWord(i, "date") ? i + 1 : i + 2);

            var n = Node.Create(NodeKind.Date);
            n.LexStart   = i;
            n.LexEnd     = end;
            n.Relative   = RelativeKind.Current;
            n.OffsetDays = 0;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        // ------------------------------------------------------------------ explicit ranges

        private int TryExplicitDateRange(int i, out int node)
        {
            node = Node.Unspecified;

            int  at         = i;
            bool sawFrom    = false;
            bool sawBetween = false;

            if (AtTerm(at, TermKind.RangeStart, out int rangeKind))
            {
                sawFrom    = rangeKind == 0;
                sawBetween = rangeKind == 1;
                at         = After(at);
                at         = SkipArticle(at);
            }

            int leftEnd = TryRangeEndpoint(at, out int left);
            if (leftEnd < 0) return -1;

            int mid = leftEnd;
            if (At(mid, LexKind.Comma)) mid++;

            bool connector = false;

            if (sawBetween && AtWord(mid, "and"))                       { connector = true; mid++; }
            else if (AtTerm(mid, TermKind.Connector) && !AtWord(mid, "and")) { connector = true; mid = After(mid); }
            else if (At(mid, LexKind.Dash) || At(mid, LexKind.Tilde))   { connector = true; mid++; }

            if (!connector) return -1;

            mid = SkipArticle(mid);

            int rightEnd = TryRangeEndpoint(mid, out int right);
            if (rightEnd < 0) return -1;

            // A bare "A - B" of two unrelated things is only a range when both sides parsed as dates
            if (!sawFrom && !sawBetween && left == Node.Unspecified) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.LexEnd   = rightEnd;
            n.Left     = left;
            n.Right    = right;
            SetSpan(ref n);
            node = Alloc(n);
            return rightEnd;
        }

        // ------------------------------------------------------------------ "before 2000", "since august", "end of this year"

        private int TryModDatePeriod(int i, out int node)
        {
            node = Node.Unspecified;

            var mod = ModKind.None;

            if (AtTerm(i, TermKind.Mod, out int modValue))
            {
                mod = (ModKind)modValue;
            }
            else if (AtTerm(i, TermKind.RangeStart, out int rangeKind) && rangeKind == 0 && !AtWord(i, "from"))
            {
                mod = ModKind.Since;   // "starting january 7th", "beginning on january 7th"
            }
            else
            {
                return -1;
            }

            if (mod == ModKind.OrLater || mod == ModKind.OrEarlier || mod == ModKind.Less || mod == ModKind.More) return -1;

            int at = After(i);

            if (mod == ModKind.Later || mod == ModKind.Earlier)
            {
                at = SkipWords(at, "in", "on");
                at = SkipArticle(at);
            }

            at = SkipWords(at, "the", "of");
            at = SkipWords(at, "the", "on");
            at = SkipArticle(at);
            if (At(at, LexKind.Dash)) at++;   // "mid-november"

            // "> = 2019", "< =2019", "=2019"
            if (At(at, LexKind.Equal)) at++;

            int inner = TryRangeEndpoint(at, out int child);

            if (inner < 0 && AtTerm(at, TermKind.Unit, out int bareUnit))
            {
                var bare = Node.Create(NodeKind.DateRange);
                bare.LexStart    = at;
                bare.LexEnd      = at + 1;
                bare.PeriodUnit  = (TimeUnit)bareUnit;
                bare.PeriodCount = 1;
                bare.Relative    = RelativeKind.This;
                SetSpan(ref bare);
                child = Alloc(bare);
                inner = at + 1;
            }

            if (inner < 0) return -1;

            ref var target = ref NodeAt(child);

            // "end of tomorrow" / "end of this sunday" name a moment, not a period
            if (mod == ModKind.End && target.Kind == NodeKind.Date) return -1;

            var n = target;
            n.LexStart = i;
            n.LexEnd   = inner;

            // "after mid may" narrows first and bounds second, so both modifiers have to survive
            if (IsBounding(mod) && IsNarrowing(target.Mod)) n.InnerMod = target.Mod;

            n.Mod = mod;

            if (n.Kind == NodeKind.Date)
            {
                // Naming a single day with "beginning"/"as late as" opens a period at that day
                if (mod == ModKind.Start || mod == ModKind.Early) n.Mod = ModKind.Since;
                if (mod == ModKind.Late)                          n.Mod = ModKind.Until;

                n.Kind = NodeKind.DateRange;
            }

            SetSpan(ref n);
            node = Alloc(n);
            return inner;
        }

        // ------------------------------------------------------------------ "first week of 2015", "the last 3 weeks of this year"

        private int TryNthPeriodOf(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);

            int  ordinal = Node.Unspecified;
            int  count   = 1;
            bool fromEnd = false;

            if (AtTermValue(at, TermKind.Relative, (int)RelativeKind.Last) || AtTermValue(at, TermKind.Relative, (int)RelativeKind.Previous))
            {
                fromEnd = true;
                at      = After(at);

                if (TryInteger(at, out int lastCount, out int afterLastCount) && lastCount > 0 && lastCount < 100)
                {
                    count = lastCount;
                    at    = afterLastCount;
                }

                ordinal   = 1;
                fromEnd   = true;
            }
            else if (AtTermValue(at, TermKind.Ordinal, 1) && TryInteger(at + 1, out int firstCount, out int afterFirstCount) && firstCount > 0 && firstCount < 100)
            {
                // "the first 2 weeks of 2021", "first ten days of last year"
                ordinal = 1;
                count   = firstCount;
                at      = afterFirstCount;
            }
            else if (TryOrdinal(at, out int ord, out int afterOrd) && ord > 0 && ord <= 60)
            {
                ordinal = ord;
                at      = afterOrd;
            }
            else
            {
                return -1;
            }

            bool business = false;
            if (AtTerm(at, TermKind.BusinessDay)) { business = true; at++; }

            if (!AtTerm(at, TermKind.Unit, out int unitValue)) return -1;

            var unit = (TimeUnit)unitValue;
            if (unit == TimeUnit.Day && count == 1) return -1;   // "the 15th day of next month" names a day

            at++;

            int ofAt = at;
            ofAt = SkipWords(ofAt, "of", "in");
            ofAt = SkipArticle(ofAt);

            int anchorEnd = TryRangeEndpoint(ofAt, out int anchor);
            if (anchorEnd < 0) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart        = i;
            n.LexEnd          = anchorEnd;
            n.PeriodUnit      = unit;
            n.PeriodCount     = count;
            n.OrdinalInPeriod = ordinal;
            n.OrdinalFromEnd  = fromEnd;
            n.Left            = anchor;
            n.BusinessDays    = business;
            SetSpan(ref n);
            node = Alloc(n);
            return anchorEnd;
        }

        // ------------------------------------------------------------------ "2 weeks starting may 20th", "within 9 months"

        private int TryDurationFromDate(int i, out int node)
        {
            node = Node.Unspecified;

            int  at      = i;
            bool forward = true;
            bool sawFor  = AtWord(at, "for");

            at = SkipWord(at, "for");

            bool within = AtWord(at, "within");
            if (within)
            {
                at++;
                at = SkipArticle(at);
                if (AtTermValue(at, TermKind.Relative, (int)RelativeKind.Next)) at++;
            }

            int durationEnd = TryDuration(at, out int duration);
            if (durationEnd < 0) return -1;

            ref var d = ref NodeAt(duration);
            if (!d.Duration.IsDateOnly && !within) return -1;

            int anchor    = Node.Unspecified;
            int end       = durationEnd;
            int startAt   = durationEnd;

            if (At(startAt, LexKind.Comma)) startAt++;

            if (AtWord(startAt, "starting") || AtWord(startAt, "beginning") || AtWord(startAt, "commencing") || (sawFor && AtWord(startAt, "from")))
            {
                int afterStart = After(startAt);
                afterStart = SkipWords(afterStart, "from", "on");
                afterStart = SkipArticle(afterStart);

                int anchorEnd = TryRangeEndpoint(afterStart, out anchor);
                if (anchorEnd < 0) return -1;

                end = anchorEnd;
            }
            else if (!within)
            {
                return -1;
            }

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = sawFor && anchor >= 0 && !At(durationEnd, LexKind.Comma) ? i + 1 : i;
            n.LexEnd   = end;
            n.Left     = duration;
            n.Anchor   = anchor;
            n.Relative = forward ? RelativeKind.Next : RelativeKind.Last;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        // ------------------------------------------------------------------ simple periods

        private int TrySimplePeriod(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TryRelativeUnitPeriod(i, out int n1), n1, ref best, ref bestNode);
            Consider(TryQuarterPeriod(i, out int n2),      n2, ref best, ref bestNode);
            Consider(TryWeekNumberPeriod(i, out int n3),   n3, ref best, ref bestNode);
            Consider(TryDecadePeriod(i, out int n4),       n4, ref best, ref bestNode);
            Consider(TryCenturyPeriod(i, out int n5),      n5, ref best, ref bestNode);
            Consider(TrySeasonPeriod(i, out int n6),       n6, ref best, ref bestNode);
            Consider(TryFiscalYearPeriod(i, out int n7),   n7, ref best, ref bestNode);
            Consider(TryMonthPeriod(i, out int n8),        n8, ref best, ref bestNode);
            Consider(TryYearPeriod(i, out int n9),         n9, ref best, ref bestNode);

            node = bestNode;
            return best;
        }

        /// <summary>"last week", "next 3 days", "the weekend", "previous 4 business days", "2 upcoming months".</summary>
        private int TryRelativeUnitPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int start    = i;
            int at       = SkipArticle(i);
            bool hadThe  = at != i;
            var relative = RelativeKind.None;
            int count    = Node.Unspecified;

            // "5 past years", "2 next days", "10 previous weeks"
            bool countLedTheUnit = false;

            if (TryInteger(at, out int leadingCount, out int afterLeading) && leadingCount > 0 && leadingCount < 1000 && AtTerm(afterLeading, TermKind.Relative))
            {
                count           = leadingCount;
                at              = afterLeading;
                countLedTheUnit = true;
            }

            if (AtTerm(at, TermKind.Relative, out int relValue))
            {
                relative = (RelativeKind)relValue;
                at       = After(at);
            }

            if (count < 0 && TryInteger(at, out int trailingCount, out int afterTrailing) && trailingCount > 0 && trailingCount < 1000)
            {
                count = trailingCount;
                at    = afterTrailing;
            }
            else if (count < 0 && AtTerm(at, TermKind.Several, out int severalCount))
            {
                count = severalCount;
                at++;
            }

            bool business = false;
            if (AtTerm(at, TermKind.BusinessDay))
            {
                business = true;
                at++;
            }

            if (!AtTerm(at, TermKind.Unit, out int unitValue)) return -1;

            // "3 next week" is the number three beside "next week", not three weeks
            if (countLedTheUnit && count > 1 && !LooksPlural(at)) return -1;

            // Romance languages put the qualifier after the unit: "la semaine prochaine"
            if (_lexicon.RelativeAfterUnit && relative == RelativeKind.None && AtTerm(at + 1, TermKind.Relative, out int trailingRel))
            {
                relative = (RelativeKind)trailingRel;

                var n2 = Node.Create(NodeKind.DateRange);
                n2.LexStart     = hadThe && (count < 0 || _lexicon.ArticleInPeriodSpan) ? i : start;
                n2.LexEnd       = at + 2;
                n2.PeriodUnit   = (TimeUnit)unitValue;
                n2.PeriodCount  = count < 0 ? 1 : count;
                n2.Relative     = relative;
                n2.BusinessDays = business;
                SetSpan(ref n2);
                node = Alloc(n2);
                return n2.LexEnd;
            }

            // "last three weekends" is the verb "last"; a period written that way says "the"
            if (!hadThe && count >= 0 && (relative == RelativeKind.Last || relative == RelativeKind.Previous) && !AtNumber(i)) return -1;

            var unit = (TimeUnit)unitValue;
            if (business && unit == TimeUnit.Day) unit = TimeUnit.BusinessDay;

            at++;

            // "the weekend" and "weekend" name a period; "three weekends" is how long something lasts
            if (relative == RelativeKind.None && (unit != TimeUnit.Weekend || count >= 0) && (!hadThe || count >= 0)) return -1;
            if (unit == TimeUnit.Hour || unit == TimeUnit.Minute || unit == TimeUnit.Second) return -1;
            if (unit == TimeUnit.Day && count < 0) return -1;   // "the day" and "next day" name a day, not a period

            if (hadThe && !_lexicon.ArticleInPeriodSpan && unit != TimeUnit.Decade && unit != TimeUnit.Century && (relative != RelativeKind.None || count >= 0)) start = i + 1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart     = start;
            n.LexEnd       = at;
            n.PeriodUnit   = unit;
            n.PeriodCount  = count < 0 ? 1 : count;
            n.Relative     = relative == RelativeKind.None ? RelativeKind.This : relative;
            n.BusinessDays = business;

            // Only "same" leaves the period unanchored; "current" names the one the reference sits in
            if (relative == RelativeKind.Current && IsSameWord(at - 1)) n.Mod = ModKind.RefUndef;
            SetSpan(ref n);
            node = Alloc(n);
            return at;
        }

        private static bool IsBounding(ModKind mod)  => mod is ModKind.Before or ModKind.After or ModKind.Since or ModKind.Until;

        private static bool IsNarrowing(ModKind mod) => mod is ModKind.Start or ModKind.Mid or ModKind.End or ModKind.Early or ModKind.Late;

        /// <summary>True when the relative word at <paramref name="i"/> is the "same" of "the same week".</summary>
        private readonly bool IsSameWord(int i)
        {
            for (int k = i; k >= 0 && k > i - 3; k--)
            {
                if (AtWord(k, "same") || AtWord(k, "selbe") || AtWord(k, "selben") || AtWord(k, "même") || AtWord(k, "meme")
                    || AtWord(k, "mismo") || AtWord(k, "misma") || AtWord(k, "mesmo") || AtWord(k, "mesma")
                    || AtWord(k, "stesso") || AtWord(k, "stessa") || AtWord(k, "zelfde") || AtWord(k, "dezelfde")) return true;
            }

            return false;
        }

        /// <summary>"q1", "2019 q1", "q3 2019", "1st quarter of 2013", "2019 h2".</summary>
        private int TryQuarterPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at    = SkipArticle(i);
            if (AtTerm(at, TermKind.QuarterMarker)) i = at;
            int year  = Node.Unspecified;
            int index = Node.Unspecified;
            int perYear = 4;

            if (TryYear(at, out int leadYear, out int afterYear))
            {
                int probe = afterYear;
                if (At(probe, LexKind.Dash)) probe++;

                if (TryQuarterMarker(probe, out index, out perYear, out int markerEnd))
                {
                    year = leadYear;
                    at   = markerEnd;
                }
                else
                {
                    return -1;
                }
            }
            else if (TryOrdinal(at, out int ordinal, out int afterOrdinal) && ordinal >= 1 && ordinal <= 4 && AtTermValue(afterOrdinal, TermKind.Unit, (int)TimeUnit.Quarter))
            {
                index = ordinal;
                at    = afterOrdinal + 1;

                int ofAt = SkipWords(at, "of", "in");
                ofAt = SkipArticle(ofAt);

                if (TryYear(ofAt, out int tailYear, out int tailEnd))
                {
                    year = tailYear;
                    at   = tailEnd;
                }
                else if (AtTermValue(ofAt, TermKind.Unit, (int)TimeUnit.Year))
                {
                    at = ofAt + 1;
                }
            }
            else if (TryQuarterMarker(at, out index, out perYear, out int markerEnd2))
            {
                at = markerEnd2;

                int probe = at;
                if (At(probe, LexKind.Dash)) probe++;
                probe = SkipWords(probe, "of", "in");

                if (TryYear(probe, out int tailYear, out int tailEnd))
                {
                    year = tailYear;
                    at   = tailEnd;
                }
            }
            else
            {
                return -1;
            }

            if (index < 1) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.LexEnd   = at;
            n.Year     = year;

            if (perYear == 4) { n.Quarter    = index; }
            else              { n.HalfOfYear = index; }

            SetSpan(ref n);
            node = Alloc(n);
            return at;
        }

        private readonly bool TryQuarterMarker(int i, out int index, out int perYear, out int end)
        {
            index   = Node.Unspecified;
            perYear = 4;
            end     = i;

            if (AtTerm(i, TermKind.QuarterMarker, out int marker))
            {
                // "q1" / "h2" — the digit has to be glued to the letter
                if (AtNumber(i + 1) && !_lex[i + 1].SpaceBefore && DigitsAt(i + 1) == 1)
                {
                    int v = NumberAt(i + 1);

                    if (v >= 1 && v <= marker)
                    {
                        index   = v;
                        perYear = marker;
                        end     = i + 2;
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>"week 23", "week 3 of 2027", "week 27 of last year".</summary>
        private int TryWeekNumberPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);

            if (!AtTermValue(at, TermKind.Unit, (int)TimeUnit.Week)) return -1;
            if (!_lexicon.ArticleInPeriodSpan) i = at;   // "the week 31" is reported as "week 31"

            at++;

            if (!AtNumber(at) || NumberAt(at) < 1 || NumberAt(at) > 53) return -1;

            int week = NumberAt(at);
            at++;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart   = i;
            n.WeekOfYear = week;

            int ofAt = SkipWords(at, "of", "in");
            ofAt = SkipArticle(ofAt);

            if (TryYear(ofAt, out int year, out int yearEnd))
            {
                n.Year = year;
                at     = yearEnd;
            }
            else if (AtTerm(ofAt, TermKind.Relative, out int relValue) && AtTermValue(ofAt + 1, TermKind.Unit, (int)TimeUnit.Year))
            {
                n.Relative     = (RelativeKind)relValue;
                n.OffsetYears  = WeekShiftOf((RelativeKind)relValue);
                at             = ofAt + 2;
            }

            n.LexEnd = at;
            SetSpan(ref n);
            node = Alloc(n);
            return at;
        }

        /// <summary>"1990s", "1990 s", "90 s", "nineties", "the next two decades".</summary>
        private int TryDecadePeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int worded = SkipArticle(i);

            if (AtTerm(worded, TermKind.Decade, out int namedDecade))
            {
                var nd = Node.Create(NodeKind.DateRange);
                nd.LexStart = i;
                nd.LexEnd   = worded + 1;
                nd.Decade   = namedDecade;
                nd.Century  = 1;   // the century is not written, so both readings stand
                SetSpan(ref nd);
                node = Alloc(nd);
                return nd.LexEnd;
            }

            if (AtNumber(i) && (DigitsAt(i) == 4 || DigitsAt(i) == 2) && NumberAt(i) % 10 == 0)
            {
                int at = i + 1;

                if (!AtWord(at, "s")) return -1;

                at++;

                int decade = NumberAt(i);

                var n = Node.Create(NodeKind.DateRange);
                n.LexStart = i;
                n.LexEnd   = at;

                if (DigitsAt(i) == 4)
                {
                    n.Decade = decade;
                }
                else
                {
                    // "90 s" names a decade of any century, so both readings are kept
                    n.Decade  = decade < 30 ? 2000 + decade : 1900 + decade;
                    n.Century = 1;
                }

                SetSpan(ref n);
                node = Alloc(n);
                return at;
            }

            return -1;
        }

        /// <summary>"21st century", "15th century".</summary>
        private int TryCenturyPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);

            if (!TryOrdinal(at, out int ordinal, out int end) || ordinal < 1 || ordinal > 30) return -1;
            if (!AtTermValue(end, TermKind.Unit, (int)TimeUnit.Century)) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.LexEnd   = end + 1;
            n.Century  = ordinal;
            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        /// <summary>"summer", "this summer", "summer of 2019".</summary>
        private int TrySeasonPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = SkipArticle(i);
            var rel = RelativeKind.None;

            if (AtTerm(at, TermKind.Relative, out int relValue))
            {
                rel = (RelativeKind)relValue;
                at  = After(at);
            }

            if (!AtTerm(at, TermKind.Season, out int seasonValue)) return -1;

            int end = at + 1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.Season   = (SeasonKind)seasonValue;
            n.Relative = rel;

            int ofAt = SkipWords(end, "of", "in");
            if (TryYear(ofAt, out int year, out int yearEnd))
            {
                n.Year = year;
                end    = yearEnd;
            }
            else if (rel != RelativeKind.None)
            {
                n.Relative = rel;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"fiscal year 2008", "cy 2008", "this school year", "cy18".</summary>
        private int TryFiscalYearPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = SkipArticle(i);
            i       = at;
            var rel = RelativeKind.None;

            if (AtTerm(at, TermKind.Relative, out int relValue))
            {
                rel = (RelativeKind)relValue;
                at  = After(at);
            }

            if (!AtTerm(at, TermKind.Fiscal, out int fiscalKind)) return -1;

            int end = at + 1;

            // "cy18" / "sy18" — the digits are glued to the two-letter form
            if (AtNumber(end) && !_lex[end].SpaceBefore && DigitsAt(end) == 2)
            {
                var n2 = Node.Create(NodeKind.DateRange);
                n2.LexStart   = i;
                n2.LexEnd     = end + 1;
                n2.FiscalKind = fiscalKind;
                n2.Year       = ExpandTwoDigitYear(NumberAt(end));
                SetSpan(ref n2);
                node = Alloc(n2);
                return n2.LexEnd;
            }

            if (AtTermValue(end, TermKind.Unit, (int)TimeUnit.Year)) end++;
            else if (fiscalKind >= 0 && !AtNumber(end)) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart   = i;
            n.FiscalKind = fiscalKind;
            n.Relative   = rel;

            if (TryYear(end, out int year, out int yearEnd))
            {
                n.Year = year;
                end    = yearEnd;
            }
            else if (rel == RelativeKind.None)
            {
                n.Relative = RelativeKind.This;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"april", "april 2017", "dec-2018", "2015-12", "june of 1992", "2017 april".</summary>
        private int TryMonthPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = SkipArticle(i);
            if (!_lexicon.ArticleInPeriodSpan) i = at;   // English reports "the april 2017" as "april 2017"
            var rel = RelativeKind.None;

            if (AtTerm(at, TermKind.Relative, out int relValue) && AtTerm(at + 1, TermKind.Month))
            {
                rel = (RelativeKind)relValue;
                at  = After(at);
            }

            // "2015-12", "2017 april", "2015-3"
            if (TryYear(at, out int leadYear, out int afterLeadYear))
            {
                int probe = afterLeadYear;
                if (At(probe, LexKind.Dash) || At(probe, LexKind.Slash)) probe++;

                if (AtTerm(probe, TermKind.Month, out int namedMonth))
                {
                    var ny = Node.Create(NodeKind.DateRange);
                    ny.LexStart = i;
                    ny.LexEnd   = probe + 1;
                    ny.Year     = leadYear;
                    ny.Month    = namedMonth;
                    SetSpan(ref ny);
                    node = Alloc(ny);
                    return ny.LexEnd;
                }

                if (AtNumber(probe) && DigitsAt(probe) <= 2 && NumberAt(probe) >= 1 && NumberAt(probe) <= 12 && probe != afterLeadYear)
                {
                    var ny = Node.Create(NodeKind.DateRange);
                    ny.LexStart = i;
                    ny.LexEnd   = probe + 1;
                    ny.Year     = leadYear;
                    ny.Month    = NumberAt(probe);
                    SetSpan(ref ny);
                    node = Alloc(ny);
                    return ny.LexEnd;
                }

                return -1;
            }

            if (!AtTerm(at, TermKind.Month, out int month))
            {
                // "12-2015"
                if (AtNumber(at) && DigitsAt(at) <= 2 && NumberAt(at) >= 1 && NumberAt(at) <= 12
                    && (At(at + 1, LexKind.Dash) || At(at + 1, LexKind.Slash)) && TryYear(at + 2, out int pairedYear, out int pairedEnd))
                {
                    var nm = Node.Create(NodeKind.DateRange);
                    nm.LexStart = i;
                    nm.LexEnd   = pairedEnd;
                    nm.Year     = pairedYear;
                    nm.Month    = NumberAt(at);
                    SetSpan(ref nm);
                    node = Alloc(nm);
                    return pairedEnd;
                }

                return -1;
            }

            int end = at + 1;

            // Only an abbreviation carries a full stop: "dec." but not "april."
            if (At(end, LexKind.Dot) && !_lex[end].SpaceBefore && _lex[at].Length <= 4 && (AtNumber(end + 1) || At(end + 1, LexKind.Word))) end++;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.Month    = month;
            n.Relative = rel;

            int yearAt   = end;
            bool spelled = false;

            if (At(yearAt, LexKind.Comma)) { yearAt++; spelled = true; }
            if (At(yearAt, LexKind.Dash) || At(yearAt, LexKind.Slash)) yearAt++;
            int beforeWords = yearAt;
            yearAt = SkipWords(yearAt, "of", "in");
            yearAt = SkipArticle(yearAt);
            if (yearAt != beforeWords) spelled = true;

            if (TryYear(yearAt, out int year, out int yearEnd))
            {
                n.Year = year;
                end    = yearEnd;
            }
            else if (spelled && AtNumber(yearAt) && DigitsAt(yearAt) == 2 && yearAt != end)
            {
                n.Year = ExpandTwoDigitYear(NumberAt(yearAt));
                end    = yearAt + 1;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>"2019", "year 2008", "1865".</summary>
        private int TryYearPeriod(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticle(i);
            bool marked = false;

            if (AtTermValue(at, TermKind.Unit, (int)TimeUnit.Year) && !AtTerm(at, TermKind.Relative))
            {
                marked = true;
                if (!_lexicon.ArticleInPeriodSpan) i = at;   // "the year 2008" is reported as "year 2008"
                at++;
            }

            if (!TryYear(at, out int year, out int end)) return -1;

            // A bare four-digit number is only a year when nothing else claims it
            if (!marked && AtNumber(at) && (At(end, LexKind.Colon) || At(end, LexKind.Slash))) return -1;

            var n = Node.Create(NodeKind.DateRange);
            n.LexStart = i;
            n.LexEnd   = end;
            n.Year     = year;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }
    }
}
