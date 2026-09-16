using System;

namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        /// <summary>Matches any date form at <paramref name="i"/>, keeping the longest.</summary>
        private int TryDate(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TryOffsetDate(i, out int n1),    n1, ref best, ref bestNode);
            Consider(TrySpecialDay(i, out int n2),    n2, ref best, ref bestNode);
            Consider(TryHolidayDate(i, out int n3),   n3, ref best, ref bestNode);
            Consider(TryWeekdayDate(i, out int n4),   n4, ref best, ref bestNode);
            Consider(TryMonthNameDate(i, out int n5), n5, ref best, ref bestNode);
            Consider(TryNumericDate(i, out int n6),   n6, ref best, ref bestNode);
            Consider(TryNthDayOfDate(i, out int n7),  n7, ref best, ref bestNode);
            Consider(TryOrdinalDay(i, out int n8),    n8, ref best, ref bestNode);

            node = bestNode;
            return best;
        }

        private static void Consider(int end, int node, ref int best, ref int bestNode)
        {
            if (end > best)
            {
                best     = end;
                bestNode = node;
            }
        }

        // ------------------------------------------------------------------ today / tomorrow / ...

        private int TrySpecialDay(int i, out int node)
        {
            node = Node.Unspecified;

            if (!AtTerm(i, TermKind.SpecialDay, out int value)) return -1;

            var kind = (SpecialDayKind)value;

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.LexEnd   = After(i);

            switch (kind)
            {
                case SpecialDayKind.Today:              n.OffsetDays =  0; break;
                case SpecialDayKind.TheDay:             n.OffsetDays =  0; break;
                case SpecialDayKind.Tomorrow:           n.OffsetDays =  1; break;
                case SpecialDayKind.NextDay:            n.OffsetDays =  1; break;
                case SpecialDayKind.Yesterday:          n.OffsetDays = -1; break;
                case SpecialDayKind.PriorDay:           n.OffsetDays = -1; break;
                case SpecialDayKind.DayAfterTomorrow:   n.OffsetDays =  2; break;
                case SpecialDayKind.DayBeforeYesterday: n.OffsetDays = -2; break;
                default: return -1;   // Now / EndOfDay are datetimes, handled elsewhere
            }

            n.Relative = RelativeKind.Current;
            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        // ------------------------------------------------------------------ holidays

        private int TryHolidayDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = i;
            var rel = RelativeKind.None;

            if (AtTerm(at, TermKind.Relative, out int relValue) && !AtTerm(at, TermKind.Holiday))
            {
                rel = (RelativeKind)relValue;
                at  = After(at);
            }

            if (!AtTerm(at, TermKind.Holiday, out int holidayValue)) return -1;

            int end = After(at);

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.Holiday  = (HolidayKind)holidayValue;
            n.Relative = rel;

            // "easter 2018" / "independence day of this year" / "saint patrick 2020"
            int afterYear = end;
            afterYear = SkipWords(afterYear, "of", "in");
            afterYear = SkipWord(afterYear, "the");

            if (TryYear(afterYear, out int year, out int yearEnd))
            {
                n.Year = year;
                end    = yearEnd;
            }
            else if (AtTerm(afterYear, TermKind.Relative, out int yearRel) && AtTermValue(afterYear + 1, TermKind.Unit, (int)TimeUnit.Year))
            {
                // "independence day of this year" names one year, so it names one day
                n.Relative = (RelativeKind)yearRel;
                end        = afterYear + 2;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        // ------------------------------------------------------------------ weekday-anchored dates

        private int TryWeekdayDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at       = SkipArticleOfDate(i, out i);
            var relative = RelativeKind.None;
            int weekShift = 0;
            bool sawWeek  = false;

            // "next week (on) monday", "this week monday", "previous week - monday"
            if (AtTerm(at, TermKind.Relative, out int leadRel) && AtTermValue(at + 1, TermKind.Unit, (int)TimeUnit.Week))
            {
                relative  = (RelativeKind)leadRel;
                weekShift = WeekShiftOf(relative);
                sawWeek   = true;
                at       += 2;
                at        = SkipWords(at, "on", "of");
                if (At(at, LexKind.Dash) || At(at, LexKind.Comma)) at++;
                at        = SkipWord(at, "on");
            }
            else
            {
                // "this past wednesday", "next friday", "coming thursday"
                while (AtTerm(at, TermKind.Relative, out int relValue) && !AtTerm(at, TermKind.Weekday))
                {
                    var r = (RelativeKind)relValue;
                    if (relative == RelativeKind.None || relative == RelativeKind.This) relative = r;
                    at = After(at);
                }
            }

            if (!AtTerm(at, TermKind.Weekday, out int weekday)) return -1;

            int end = at + 1;

            // Only an abbreviation carries a full stop: "mer." but not "tuesday."
            if (At(end, LexKind.Dot) && !_lex[end].SpaceBefore && _lex[at].Length <= 4 && (AtNumber(end + 1) || At(end + 1, LexKind.Word))) end++;

            var n = Node.Create(NodeKind.Date);
            n.LexStart    = i;
            n.Weekday     = weekday;
            n.Relative    = relative;
            n.OffsetWeeks = weekShift;

            if (!sawWeek)
            {
                // "tuesday of next week", "on monday of the following week"
                int probe = end;
                probe = SkipWord(probe, "of");
                probe = SkipArticle(probe);

                if (AtTerm(probe, TermKind.Relative, out int tailRel) && AtTermValue(probe + 1, TermKind.Unit, (int)TimeUnit.Week))
                {
                    n.Relative    = (RelativeKind)tailRel;
                    n.OffsetWeeks = WeekShiftOf((RelativeKind)tailRel);
                    end           = probe + 2;
                    n.LexEnd      = end;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return end;
                }
            }

            // "friday 5/12", "tuesday march 7", "friday 2018-7-6"
            int tail = end;
            tail = SkipGlue(tail, 1);
            if (At(tail, LexKind.Comma)) tail++;
            if (At(tail, LexKind.Dash) && sawWeek) tail++;

            int dateEnd = TryMonthNameDate(tail, out int attached);
            if (dateEnd < 0) dateEnd = TryNumericDate(tail, out attached);

            if (dateEnd > 0)
            {
                ref var a = ref NodeAt(attached);
                n.Year  = a.Year;
                n.Month = a.Month;
                n.Day   = a.Day;
                end     = dateEnd;
            }
            else
            {
                // "fri 14th", "tuesday the eleventh", "monday 21"
                int probe   = tail;
                bool theDay = false;

                for (int k = end; k <= probe; k++)
                {
                    if (AtWord(k, "the")) { theDay = true; break; }
                }

                probe = SkipArticle(probe);
                if (AtWord(tail, "the")) theDay = true;

                if (TryOrdinal(probe, out int ord, out int ordEnd) && ord >= 1 && ord <= 31)
                {
                    n.Day         = ord;
                    n.DefiniteDay = theDay;
                    end           = ordEnd;
                }
                else if (AtNumber(probe) && NumberAt(probe) >= 1 && NumberAt(probe) <= 31 && DigitsAt(probe) <= 2 && _lex[probe].SpaceBefore
                         && !AtTerm(probe + 1, TermKind.AmPm))
                {
                    // "mon 9 am" is nine o'clock on a monday, not the ninth
                    n.Day = NumberAt(probe);
                    end   = probe + 1;
                }
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        /// <summary>A weekday with no date glued to it, so "monday 8-9am" can read 8-9 as a clock range.</summary>
        private int TryWeekdayBare(int i, out int node)
        {
            node = Node.Unspecified;

            int at  = i;
            var rel = RelativeKind.None;

            while (AtTerm(at, TermKind.Relative, out int relValue) && !AtTerm(at, TermKind.Weekday))
            {
                if (rel == RelativeKind.None || rel == RelativeKind.This) rel = (RelativeKind)relValue;
                at = After(at);
            }

            if (!AtTerm(at, TermKind.Weekday, out int weekday)) return -1;

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.LexEnd   = at + 1;
            n.Weekday  = weekday;
            n.Relative = rel;
            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        private static int WeekShiftOf(RelativeKind relative) => relative switch
        {
            RelativeKind.Next      =>  1,
            RelativeKind.Coming    =>  1,
            RelativeKind.Following =>  1,
            RelativeKind.Last      => -1,
            RelativeKind.Previous  => -1,
            RelativeKind.JustPast  => -1,
            _                      =>  0,
        };

        // ------------------------------------------------------------------ month-name dates

        /// <summary>A day of the month written as digits, an ordinal or a spelled-out number.</summary>
        private readonly bool TryDayNumber(int i, out int day, out int end)
        {
            if (TryOrdinal(i, out day, out end) && day >= 1 && day <= 31) return true;

            if (AtNumber(i) && DigitsAt(i) <= 2 && NumberAt(i) >= 1 && NumberAt(i) <= 31)
            {
                day = NumberAt(i);
                end = i + 1;
                return true;
            }

            if (TryWordNumber(i, out int words, out int wordsEnd) && words >= 1 && words <= 31)
            {
                day = words;
                end = wordsEnd;
                return true;
            }

            day = Node.Unspecified;
            end = i;
            return false;
        }

        /// <summary>A year in any accepted spelling, including two digits.</summary>
        private readonly bool TryYearLoose(int i, out int year, out int end)
        {
            if (TryYear(i, out year, out end)) return true;

            if (AtNumber(i) && DigitsAt(i) == 2 && !At(i + 1, LexKind.Colon))
            {
                year = ExpandTwoDigitYear(NumberAt(i));
                end  = i + 1;
                return true;
            }

            return false;
        }

        private int TryMonthNameDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticleOfDate(i, out i);

            // ---- "<month> <day> [, <year>]"
            if (AtTerm(at, TermKind.Month, out int month))
            {
                int afterMonth = at + 1;
                if (At(afterMonth, LexKind.Dot) && !_lex[afterMonth].SpaceBefore)   afterMonth++;
                if (At(afterMonth, LexKind.Comma))                                  afterMonth++;
                if ((At(afterMonth, LexKind.Slash) || At(afterMonth, LexKind.Dash)) && !_lex[afterMonth].SpaceBefore) afterMonth++;
                afterMonth = SkipArticle(afterMonth);

                // Reject "april 2017" and "december" — those are month periods, not dates
                if (TryDayNumber(afterMonth, out int day, out int dayEnd))
                {
                    var n = Node.Create(NodeKind.Date);
                    n.LexStart = i;
                    n.Month    = month;
                    n.Day      = day;

                    int end = dayEnd;
                    int yearAt = end;
                    if (At(yearAt, LexKind.Comma)) yearAt++;
                    if ((At(yearAt, LexKind.Slash) || At(yearAt, LexKind.Dash)) && !_lex[yearAt].SpaceBefore) yearAt++;
                    yearAt = SkipGlue(yearAt);

                    if (TryYearLoose(yearAt, out int year, out int yearEnd))
                    {
                        n.Year = year;
                        end    = yearEnd;
                    }

                    n.LexEnd = end;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return end;
                }
            }

            // ---- "<day> [of] <month> [<year>]"
            if (TryDayNumber(at, out int day2, out int day2End))
            {
                int afterDay = day2End;
                afterDay = SkipWord(afterDay, "day");
                afterDay = SkipGlue(afterDay);
                if ((At(afterDay, LexKind.Slash) || At(afterDay, LexKind.Dash)) && !_lex[afterDay].SpaceBefore) afterDay++;

                if (AtTerm(afterDay, TermKind.Month, out int month2))
                {
                    var n = Node.Create(NodeKind.Date);
                    n.LexStart = i;
                    n.Month    = month2;
                    n.Day      = day2;

                    int end    = afterDay + 1;
                    int yearAt = end;
                    if (At(yearAt, LexKind.Dot) && !_lex[yearAt].SpaceBefore) yearAt++;
                    if (At(yearAt, LexKind.Comma)) yearAt++;
                    if ((At(yearAt, LexKind.Slash) || At(yearAt, LexKind.Dash)) && !_lex[yearAt].SpaceBefore) yearAt++;
                    yearAt = SkipGlue(yearAt);

                    if (TryYearLoose(yearAt, out int year2, out int year2End))
                    {
                        n.Year = year2;
                        end    = year2End;
                    }

                    n.LexEnd = end;
                    SetSpan(ref n);
                    node = Alloc(n);
                    return end;
                }
            }

            return -1;
        }

        // ------------------------------------------------------------------ numeric dates

        /// <summary>One slot of a numeric date: either a number or a month name.</summary>
        private readonly struct DateSlot
        {
            public readonly int  Number;
            public readonly int  Digits;
            public readonly int  Month;    // -1 when the slot was a number

            public DateSlot(int number, int digits, int month)
            {
                Number = number;
                Digits = digits;
                Month  = month;
            }

            public bool IsMonthName => Month > 0;
        }

        private readonly bool TryDateSlot(int i, out DateSlot slot, out int end)
        {
            if (AtNumber(i))
            {
                slot = new DateSlot(NumberAt(i), DigitsAt(i), -1);
                end  = i + 1;
                return true;
            }

            if (AtTerm(i, TermKind.Month, out int month))
            {
                slot = new DateSlot(month, 0, month);
                end  = i + 1;
                return true;
            }

            slot = default;
            end  = i;
            return false;
        }

        private int TryNumericDate(int i, out int node)
        {
            node = Node.Unspecified;

            // "this 5/12" / "next 5/12" — the relative word chooses which occurrence is meant
            if (AtTerm(i, TermKind.Relative, out int leadingRelative) && !AtTerm(i, TermKind.Weekday))
            {
                int inner = TryNumericDate(After(i), out int plain);

                if (inner > 0)
                {
                    var shifted = NodeAt(plain);
                    shifted.LexStart = i;
                    shifted.Relative = (RelativeKind)leadingRelative;
                    SetSpan(ref shifted);
                    node = Alloc(shifted);
                    return inner;
                }

                return -1;
            }

            // ---- compact ISO: 20200701
            if (AtNumber(i) && DigitsAt(i) == 8)
            {
                int v  = NumberAt(i);
                int y  = v / 10000;
                int mo = (v / 100) % 100;
                int d  = v % 100;

                if (y >= 1000 && y <= 3000 && mo >= 1 && mo <= 12 && d >= 1 && d <= 31)
                {
                    var iso = Node.Create(NodeKind.Date);
                    iso.LexStart = i;
                    iso.LexEnd   = i + 1;
                    iso.Year     = y;
                    iso.Month    = mo;
                    iso.Day      = d;
                    SetSpan(ref iso);
                    node = Alloc(iso);
                    return i + 1;
                }
            }

            if (!TryDateSlot(i, out var s0, out int after0)) return -1;

            var sepKind = KindOf(after0);
            bool spaced = false;

            if (sepKind != LexKind.Slash && sepKind != LexKind.Dash && sepKind != LexKind.Dot)
            {
                // "2016 10 16" and, where the day comes first, "18 08 1978"
                bool yearFirst = s0.Digits == 4 && AtNumber(after0) && DigitsAt(after0) <= 2 && AtNumber(after0 + 1) && DigitsAt(after0 + 1) <= 2;
                bool dayFirst  = _lexicon.DayMonthOrder && s0.Digits <= 2 && AtNumber(after0) && DigitsAt(after0) <= 2
                                 && AtNumber(after0 + 1) && (DigitsAt(after0 + 1) == 4 || DigitsAt(after0 + 1) == 2)
                                 && _lex[after0].SpaceBefore && _lex[after0 + 1].SpaceBefore;

                if (yearFirst || dayFirst)
                {
                    spaced = true;
                }
                else
                {
                    return -1;
                }
            }

            int at = spaced ? after0 : after0 + 1;

            if (!TryDateSlot(at, out var s1, out int after1)) return -1;

            int end   = after1;
            bool has2 = false;
            DateSlot s2 = default;

            int sep2 = after1;

            if (spaced)
            {
                if (TryDateSlot(sep2, out s2, out int after2)) { has2 = true; end = after2; }
            }
            else if (KindOf(sep2) == sepKind && TryDateSlot(sep2 + 1, out s2, out int after2b))
            {
                has2 = true;
                end  = after2b;
            }

            int year  = Node.Unspecified;
            int month = Node.Unspecified;
            int day   = Node.Unspecified;

            if (has2)
            {
                if (!AssignThree(s0, s1, s2, ref year, ref month, ref day)) return -1;
            }
            else
            {
                if (sepKind == LexKind.Dot && !s0.IsMonthName && !s1.IsMonthName && (s0.Digits > 2 || s1.Digits > 2)) return -1;
                if (!AssignTwo(s0, s1, ref year, ref month, ref day)) return -1;
            }

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.LexEnd   = end;
            n.Year     = year;
            n.Month    = month;
            n.Day      = day;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        private readonly bool AssignThree(DateSlot a, DateSlot b, DateSlot c, ref int year, ref int month, ref int day)
        {
            Span<DateSlot> slots = stackalloc DateSlot[3] { a, b, c };

            int monthSlot = -1;
            for (int k = 0; k < 3; k++)
            {
                if (slots[k].IsMonthName)
                {
                    if (monthSlot >= 0) return false;
                    monthSlot = k;
                }
            }

            if (monthSlot >= 0)
            {
                month = slots[monthSlot].Month;

                int yearSlot = -1;
                for (int k = 0; k < 3; k++)
                {
                    if (k != monthSlot && slots[k].Digits == 4) { yearSlot = k; break; }
                }

                if (yearSlot < 0)
                {
                    // "05-aug-16" — the last remaining slot is the year
                    for (int k = 2; k >= 0; k--)
                    {
                        if (k != monthSlot) { yearSlot = k; break; }
                    }
                }

                int daySlot = -1;
                for (int k = 0; k < 3; k++)
                {
                    if (k != monthSlot && k != yearSlot) { daySlot = k; break; }
                }

                if (daySlot < 0 || yearSlot < 0) return false;

                year = ExpandTwoDigitYear(slots[yearSlot].Number);
                day  = slots[daySlot].Number;

                return day >= 1 && day <= 31 && year >= 1000 && year <= 3000;
            }

            if (a.Digits == 4)
            {
                year  = a.Number;
                month = b.Number;
                day   = c.Number;
            }
            else
            {
                year = ExpandTwoDigitYear(c.Number);

                if (_lexicon.DayMonthOrder)
                {
                    month = b.Number;
                    day   = a.Number;

                    if (month > 12 && a.Number <= 12) { month = a.Number; day = b.Number; }
                }
                else
                {
                    month = a.Number;
                    day   = b.Number;

                    if (month > 12 && b.Number <= 12) { month = b.Number; day = a.Number; }
                }
            }

            return month >= 1 && month <= 12 && day >= 1 && day <= 31 && year >= 1000 && year <= 3000;
        }

        private readonly bool AssignTwo(DateSlot a, DateSlot b, ref int year, ref int month, ref int day)
        {
            if (a.IsMonthName && !b.IsMonthName)
            {
                month = a.Month;
                day   = b.Number;
                return day >= 1 && day <= 31 && b.Digits <= 2;
            }

            if (b.IsMonthName && !a.IsMonthName)
            {
                month = b.Month;
                day   = a.Number;
                return day >= 1 && day <= 31 && a.Digits <= 2;
            }

            if (a.IsMonthName || b.IsMonthName) return false;

            // A four-digit component makes this a year/month period, not a date
            if (a.Digits == 4 || b.Digits == 4) return false;

            if (_lexicon.DayMonthOrder)
            {
                month = b.Number;
                day   = a.Number;

                if (month > 12 && a.Number <= 12) { month = a.Number; day = b.Number; }
            }
            else
            {
                month = a.Number;
                day   = b.Number;

                if (month > 12 && b.Number <= 12) { month = b.Number; day = a.Number; }
            }

            return month >= 1 && month <= 12 && day >= 1 && day <= 31;
        }

        // ------------------------------------------------------------------ "the 18th", "the 15th day of next month"

        private int TryOrdinalDay(int i, out int node)
        {
            node = Node.Unspecified;

            int at      = i;
            bool hadThe = AtWord(at, "the");
            if (hadThe) at++;

            var mod = ModKind.None;
            if (AtTerm(at, TermKind.Approx))
            {
                mod = ModKind.Approx;
                at  = After(at);

                int beforeArticle = at;
                at     = SkipArticle(at);
                hadThe = hadThe || at != beforeArticle;
            }

            bool ordinal = TryOrdinal(at, out int day, out int end);

            if (!ordinal && hadThe && AtNumber(at) && DigitsAt(at) <= 2 && NumberAt(at) >= 1 && NumberAt(at) <= 31)
            {
                day = NumberAt(at);
                end = at + 1;
            }
            else if (!ordinal || day < 1 || day > 31)
            {
                return -1;
            }

            // A bare ordinal without "the" is only a date when it is written with its suffix ("29th")
            if (!hadThe && !(AtNumber(at) && AtTerm(at + 1, TermKind.OrdinalSuffix))) return -1;

            // "3rd week of 2018" counts weeks; the ordinal belongs to the period, not to a day
            if (AtTerm(end, TermKind.Unit) || AtTerm(end, TermKind.BusinessDay)) return -1;

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.Day      = day;
            n.Mod      = mod;

            // "around the 21st this month" / "the 4th of next month"
            int tail = end;
            tail = SkipWords(tail, "of", "in");
            tail = SkipArticle(tail);

            if (AtTerm(tail, TermKind.Relative, out int relValue) && AtTermValue(tail + 1, TermKind.Unit, (int)TimeUnit.Month))
            {
                n.Relative     = (RelativeKind)relValue;
                n.OffsetMonths = WeekShiftOf((RelativeKind)relValue);
                end            = tail + 2;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        private int TryNthDayOfDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at = SkipArticleOfDate(i, out i);

            if (!TryOrdinal(at, out int ordinal, out int end)) return -1;

            if (at != i && AtTerm(at, TermKind.Cardinal)) i = at;   // "the twenty third day of september"
            if (ordinal < 1 || ordinal > 31) return -1;

            if (!AtTermValue(end, TermKind.Unit, (int)TimeUnit.Day)) return -1;

            int tail = end + 1;
            tail = SkipWords(tail, "of", "in");
            tail = SkipArticle(tail);

            var n = Node.Create(NodeKind.Date);
            n.LexStart = i;
            n.Day      = ordinal;

            if (AtTerm(tail, TermKind.Month, out int month))
            {
                n.Month  = month;
                int afterMonth = tail + 1;

                if (TryYearLoose(afterMonth, out int year, out int yearEnd))
                {
                    n.Year = year;
                    afterMonth = yearEnd;
                }

                n.LexEnd = afterMonth;
            }
            else if (AtTerm(tail, TermKind.Relative, out int relValue) && AtTermValue(tail + 1, TermKind.Unit, (int)TimeUnit.Month))
            {
                n.Relative     = (RelativeKind)relValue;
                n.OffsetMonths = WeekShiftOf((RelativeKind)relValue);
                n.LexEnd       = tail + 2;
            }
            else if (AtTermValue(tail, TermKind.Unit, (int)TimeUnit.Month))
            {
                n.LexEnd = tail + 1;
            }
            else
            {
                return -1;
            }

            SetSpan(ref n);
            node = Alloc(n);
            return n.LexEnd;
        }

        // ------------------------------------------------------------------ offsets: "2 days ago", "in two weeks", "3 days from today"

        private int TryOffsetDate(int i, out int node)
        {
            node = Node.Unspecified;

            int at         = i;
            int leadWeekday = Node.Unspecified;
            var mod         = ModKind.None;

            // "saturday 3 days from now", "thursday, two year from now", "monday two weeks from now"
            if (AtTerm(at, TermKind.Weekday, out int wd) && !AtTerm(at, TermKind.Month))
            {
                int probe = at + 1;
                if (At(probe, LexKind.Comma)) probe++;

                if (LooksLikeOffsetStart(probe))
                {
                    leadWeekday = wd;
                    at          = probe;
                }
            }

            bool sawIn = false;

            if (AtTerm(at, TermKind.Mod, out int leadMod) && ((ModKind)leadMod == ModKind.Less || (ModKind)leadMod == ModKind.More))
            {
                mod = (ModKind)leadMod;
                at  = After(at);
            }

            if (AtWord(at, "in") || AtWord(at, "within"))
            {
                if (AtWord(at + 1, "the")) return -1;   // "in the week" is a period, not an offset

                sawIn = true;
                at++;
            }

            int durationEnd = TryDuration(at, out int durationNode);
            if (durationEnd < 0) return -1;

            ref var duration = ref NodeAt(durationNode);
            var parts = duration.Duration;

            if (!parts.IsDateOnly) return -1;   // "in 3 minutes" is a datetime

            int  sign   = 0;
            int  end    = durationEnd;
            int  anchor = Node.Unspecified;
            int  tail   = durationEnd;

            if (AtTerm(tail, TermKind.Ago))
            {
                sign = -1;
                end  = After(tail);
            }
            else if (AtTerm(tail, TermKind.FromNow))
            {
                sign = 1;
                end  = After(tail);
            }
            else if (AtWord(tail, "from") || AtWord(tail, "after") || AtWord(tail, "before") || AtTerm(tail, TermKind.Mod))
            {
                bool backwards = AtWord(tail, "before") || (AtTerm(tail, TermKind.Mod, out int mv) && (ModKind)mv == ModKind.Before);
                int  anchorAt  = tail + 1;

                if (AtTerm(anchorAt, TermKind.SpecialDay, out int sd) && (SpecialDayKind)sd == SpecialDayKind.Now)
                {
                    sign = backwards ? -1 : 1;
                    end  = After(anchorAt);
                }
                else
                {
                    int anchorEnd = TryDate(anchorAt, out anchor);

                    if (anchorEnd > 0)
                    {
                        sign = backwards ? -1 : 1;
                        end  = anchorEnd;
                    }
                    else
                    {
                        return -1;
                    }
                }
            }
            else if (sawIn)
            {
                sign = 1;
            }
            else
            {
                return -1;
            }

            var n = Node.Create(mod == ModKind.None ? NodeKind.Date : NodeKind.DateRange);
            n.LexStart     = i;
            n.OffsetYears  = sign * (int)parts.Years;
            n.OffsetMonths = sign * (int)parts.Months;
            n.OffsetWeeks  = sign * (int)parts.Weeks;
            n.OffsetDays   = sign * (int)(parts.Days + parts.Weekends * 2);
            n.Anchor       = anchor;
            n.Relative     = RelativeKind.Current;

            if (mod == ModKind.More)
            {
                // "more than 2 weeks before today" is everything up to that day
                n.Mod = sign < 0 ? ModKind.Before : ModKind.After;
            }
            else if (mod == ModKind.Less)
            {
                // "less than 3 days after tomorrow" is the stretch between the two
                n.Left         = durationNode;
                n.Anchor       = anchor;
                n.OffsetYears  = 0;
                n.OffsetMonths = 0;
                n.OffsetWeeks  = 0;
                n.OffsetDays   = 0;
                n.Mod          = ModKind.None;
                n.Relative     = sign < 0 ? RelativeKind.Last : RelativeKind.Next;
            }

            // "in 3 days on friday", "two months from today on tuesday", "3 weeks after christmas on friday"
            int onAt = end;
            onAt = SkipWord(onAt, "on");

            if (AtTerm(onAt, TermKind.Weekday, out int trailingWeekday))
            {
                n.Weekday = trailingWeekday;
                end       = onAt + 1;
            }
            else if (leadWeekday >= 0)
            {
                n.Weekday = leadWeekday;
            }

            n.LexEnd = end;
            SetSpan(ref n);
            node = Alloc(n);
            return end;
        }

        private readonly bool LooksLikeOffsetStart(int i)
        {
            if (AtWord(i, "in") || AtWord(i, "within")) return true;
            if (AtNumber(i)) return true;
            if (AtTerm(i, TermKind.Cardinal)) return true;
            if (AtTerm(i, TermKind.Several)) return true;
            return false;
        }
    }
}
