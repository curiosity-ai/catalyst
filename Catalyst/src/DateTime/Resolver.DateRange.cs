using System;
using System.Collections.Generic;
using System.Globalization;

namespace Catalyst.DateTimeRecognition
{
    public sealed partial class Resolver
    {
        /// <summary>A resolved period: the half-open interval it covers and the TIMEX that names it.</summary>
        internal struct Period
        {
            public DateTime Start;
            public DateTime End;
            public string   Timex;
            public bool     NoBounds;   // seasons and fiscal years have a TIMEX but no dates
            public bool     YearUnknown;
        }

        internal bool ComputePeriod(int nodeIndex, out Period period, out Period alternate, out bool hasAlternate)
        {
            ref var n = ref At(nodeIndex);

            period       = default;
            alternate    = default;
            hasAlternate = false;

            // ---- "to date": the reference moment itself
            if (n.PresentRef)
            {
                period.Start = _reference.Date;
                period.End   = _reference.Date;
                period.Timex = "PRESENT_REF";
                return true;
            }

            // ---- explicit "A to B"
            if (n.Left >= 0 && n.Right >= 0)
            {
                if (!EndpointOf(n.Left,  out var leftStart,  out var leftTimex,  out bool leftYearUnknown))  return false;
                if (!EndpointOf(n.Right, out var rightStart, out var rightTimex, out bool rightYearUnknown)) return false;

                // "between now and november 15th" — "now" names no year, so the other end keeps its own
                bool anchoredToNow = IsNow(n.Left) || IsNow(n.Right);

                // A year written on one side applies to both: "nov-feb 2017" starts in 2016
                if (leftYearUnknown && !rightYearUnknown)
                {
                    leftStart = SafeDate(rightStart.Year, leftStart.Month, leftStart.Day);
                    if (leftStart > rightStart) leftStart = leftStart.AddYears(-1);
                    leftYearUnknown = false;
                    leftTimex       = FormatDate(leftStart);
                }
                else if (rightYearUnknown && !leftYearUnknown)
                {
                    rightStart = SafeDate(leftStart.Year, rightStart.Month, rightStart.Day);
                    if (rightStart < leftStart) rightStart = rightStart.AddYears(1);
                    rightYearUnknown = false;
                    rightTimex       = FormatDate(rightStart);
                }
                else if (rightStart < leftStart && leftYearUnknown && rightYearUnknown)
                {
                    rightStart = rightStart.AddYears(1);
                }

                bool months = (PrefersMonths(n.Left) && PrefersMonths(n.Right)) || NamesPartOfPeriod(At(n.Left).Mod) || NamesPartOfPeriod(At(n.Right).Mod);

                // "from next monday to friday" — the closing weekday is the one that comes next
                ref var right = ref At(n.Right);

                if (right.Weekday >= 0 && right.Day < 0 && right.Month < 0 && right.Year < 0 && right.Relative == RelativeKind.None)
                {
                    rightStart       = leftStart.AddDays((right.Weekday - (int)leftStart.DayOfWeek + 7) % 7);
                    rightTimex       = FormatDate(rightStart);
                    rightYearUnknown = false;
                }

                string span = SpanTimex(leftStart, rightStart, months, daysOnly: anchoredToNow);

                period.Start = leftStart;
                period.End   = rightStart;
                period.Timex = $"({leftTimex},{rightTimex},{span})";

                if (leftYearUnknown && rightYearUnknown && !anchoredToNow)
                {
                    // "from sep to nov" is this year's and last year's while this year's has not gone by
                    int shift = period.End > _reference.Date ? -1 : 0;

                    period.Start    = period.Start.AddYears(shift);
                    period.End      = period.End.AddYears(shift);
                    alternate.Start = period.Start.AddYears(1);
                    alternate.End   = period.End.AddYears(1);
                    alternate.Timex = period.Timex;
                    hasAlternate    = true;
                }

                return true;
            }

            // ---- "labor day weekend": the weekend nearest the holiday, stretched to take the holiday in
            if (n.PeriodUnit == TimeUnit.Weekend && n.Left >= 0 && At(n.Left).Holiday != HolidayKind.None)
            {
                ref var holiday = ref At(n.Left);

                if (holiday.Year >= 0)
                {
                    period = HolidayWeekend(Holidays.Resolve(holiday.Holiday, holiday.Year));
                    return true;
                }

                var thisYear = Holidays.Resolve(holiday.Holiday, _reference.Year);

                if (thisYear >= _reference.Date)
                {
                    period    = HolidayWeekend(Holidays.Resolve(holiday.Holiday, _reference.Year - 1));
                    alternate = HolidayWeekend(thisYear);
                }
                else
                {
                    period    = HolidayWeekend(thisYear);
                    alternate = HolidayWeekend(Holidays.Resolve(holiday.Holiday, _reference.Year + 1));
                }

                hasAlternate = true;
                return true;
            }

            // ---- "the week of april 10th": the week the date falls in, named by that date
            if (n.PeriodUnit == TimeUnit.Week && n.Anchor >= 0 && n.Left < 0)
            {
                if (!ComputeDate(n.Anchor, out var anchorTimex, out var anchorFirst, out var anchorSecond, out bool anchorTwo)) return false;

                period.Start = StartOfIsoWeek(anchorFirst);
                period.End   = period.Start.AddDays(7);
                period.Timex = anchorTimex;

                if (anchorTwo)
                {
                    alternate.Start = StartOfIsoWeek(anchorSecond);
                    alternate.End   = alternate.Start.AddDays(7);
                    alternate.Timex = anchorTimex;
                    hasAlternate    = true;
                }

                return true;
            }

            // ---- a duration measured from a date: "2 weeks starting may 20th", "within 9 months"
            if (n.Left >= 0 && At(n.Left).Kind == NodeKind.Duration)
            {
                var parts  = At(n.Left).Duration;
                DateTime start = _reference.Date;

                if (n.Anchor >= 0 && ComputeAnchorDateInReferenceYear(n.Anchor, out var anchorDate)) start = anchorDate;

                var end = AddParts(start, parts);

                if (n.Relative == RelativeKind.Last)
                {
                    end   = start;
                    start = AddParts(start, Negate(parts));
                }

                period.Start = start;
                period.End   = end;
                period.Timex = $"({FormatDate(start)},{FormatDate(end)},{parts.ToTimex()})";
                return true;
            }

            // ---- "first week of 2015", "the last 3 weeks of this year"
            if (n.OrdinalInPeriod > 0 && n.Left >= 0)
            {
                if (!ComputePeriod(n.Left, out var host, out var hostAlternate, out bool hostHasAlternate)) return false;

                ref var host2 = ref At(n.Left);

                period = NthOf(host, n, host2.Year < 0 && host2.Month >= 0);

                if (hostHasAlternate)
                {
                    alternate    = NthOf(hostAlternate, n, host2.Year < 0 && host2.Month >= 0);
                    hasAlternate = true;
                }

                return true;
            }

            // ---- a season
            if (n.Season != SeasonKind.None)
            {
                string code = n.Season switch
                {
                    SeasonKind.Spring => "SP",
                    SeasonKind.Summer => "SU",
                    SeasonKind.Fall   => "FA",
                    _                 => "WI",
                };

                int year = n.Year >= 0 ? n.Year : (n.Relative != RelativeKind.None ? _reference.Year + WeekShift(n.Relative) : Node.Unspecified);

                period.Timex    = year >= 0 ? $"{year:0000}-{code}" : code;
                period.NoBounds = true;
                return true;
            }

            // ---- a fiscal / calendar / school year
            if (n.FiscalKind >= 0)
            {
                int year = n.Year >= 0 ? n.Year : _reference.Year + WeekShift(n.Relative);

                if (n.FiscalKind == 0)
                {
                    period.Start = new DateTime(year, 1, 1);
                    period.End   = new DateTime(year + 1, 1, 1);
                    period.Timex = $"{year:0000}";
                    return true;
                }

                string stamp = n.Year >= 0 || n.Relative != RelativeKind.None ? $"{year:0000}" : "XXXX";
                period.Timex    = n.FiscalKind == 1 ? $"FY{stamp}" : $"SY{stamp}";
                period.NoBounds = true;
                return true;
            }

            // ---- a century
            if (n.Century > 0 && n.Decade <= 0)
            {
                int startYear = (n.Century - 1) * 100;
                period.Start  = new DateTime(startYear == 0 ? 1 : startYear, 1, 1);
                period.End    = new DateTime(startYear + 100, 1, 1);
                period.Timex  = $"({FormatDate(period.Start)},{FormatDate(period.End)},P100Y)";
                return true;
            }

            // ---- a decade
            if (n.Decade > 0)
            {
                period.Start = new DateTime(n.Decade, 1, 1);
                period.End   = new DateTime(n.Decade + 10, 1, 1);

                if (n.Century > 0)   // written without a century: "90 s", "nineties"
                {
                    int tens = n.Decade % 100;
                    period.Timex    = $"(XX{tens:00}-01-01,XX{(tens + 10) % 100:00}-01-01,P10Y)";
                    alternate.Start = period.Start.AddYears(100);
                    alternate.End   = period.End.AddYears(100);
                    alternate.Timex = period.Timex;
                    hasAlternate    = true;
                }
                else
                {
                    period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},P10Y)";
                }

                return true;
            }

            // ---- a quarter or a half-year
            if (n.Quarter > 0 || n.HalfOfYear > 0)
            {
                int months = n.Quarter > 0 ? 3 : 6;
                int index  = n.Quarter > 0 ? n.Quarter : n.HalfOfYear;

                if (n.Year >= 0)
                {
                    period.Start = new DateTime(n.Year, (index - 1) * months + 1, 1);
                    period.End   = period.Start.AddMonths(months);
                    period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},P{months}M)";
                    return true;
                }

                var thisYear = new DateTime(_reference.Year, (index - 1) * months + 1, 1);

                period.Start   = thisYear;
                period.End     = thisYear.AddMonths(months);
                period.Timex   = $"(XXXX-{(index - 1) * months + 1:00}-01,XXXX-{(((index - 1) * months + months) % 12) + 1:00}-01,P{months}M)";

                alternate.Start = thisYear.AddYears(1);
                alternate.End   = period.End.AddYears(1);
                alternate.Timex = period.Timex;
                hasAlternate    = true;
                return true;
            }

            // ---- an ISO week number
            if (n.WeekOfYear > 0)
            {
                int year = n.Year >= 0 ? n.Year : _reference.Year + n.OffsetYears;
                var start = ISOWeek.ToDateTime(year, Math.Min(n.WeekOfYear, ISOWeek.GetWeeksInYear(year)), DayOfWeek.Monday);

                period.Start = start;
                period.End   = start.AddDays(7);
                period.Timex = $"{year:0000}-W{n.WeekOfYear:00}";
                return true;
            }

            // ---- a relative run of units: "last week", "next 3 days", "the weekend"
            if (n.PeriodUnit != TimeUnit.None)
            {
                return ComputeRelativePeriod(nodeIndex, ref period);
            }

            // ---- a calendar month or year
            if (n.Month >= 0 && n.Day < 0)
            {
                if (n.Year >= 0)
                {
                    period.Start = new DateTime(n.Year, n.Month, 1);
                    period.End   = period.Start.AddMonths(1);
                    period.Timex = $"{n.Year:0000}-{n.Month:00}";
                    return true;
                }

                int baseYear = _reference.Year + WeekShift(n.Relative);
                var candidate = new DateTime(baseYear, n.Month, 1);

                period.Timex = $"XXXX-{n.Month:00}";

                if (n.Relative != RelativeKind.None)
                {
                    period.Start = candidate;
                    period.End   = candidate.AddMonths(1);
                    period.Timex = $"{baseYear:0000}-{n.Month:00}";
                    return true;
                }

                if (candidate >= _reference.Date)
                {
                    period.Start    = candidate.AddYears(-1);
                    period.End      = period.Start.AddMonths(1);
                    alternate.Start = candidate;
                    alternate.End   = candidate.AddMonths(1);
                }
                else
                {
                    period.Start    = candidate;
                    period.End      = candidate.AddMonths(1);
                    alternate.Start = candidate.AddYears(1);
                    alternate.End   = alternate.Start.AddMonths(1);
                }

                alternate.Timex = period.Timex;
                hasAlternate    = true;
                return true;
            }

            if (n.Year >= 0 && n.Day < 0)
            {
                period.Start = new DateTime(n.Year, 1, 1);
                period.End   = new DateTime(n.Year + 1, 1, 1);
                period.Timex = $"{n.Year:0000}";
                return true;
            }

            // ---- a single date widened into a period ("the week of april 10th", "before january 1, 2007")
            if (ComputeDate(nodeIndex, out var dateTimex, out var d1, out var d2, out bool twoDates))
            {
                period.Start = d1;
                period.End   = d1.AddDays(1);
                period.Timex = dateTimex;
                period.YearUnknown = dateTimex is object && dateTimex.StartsWith("XXXX", StringComparison.Ordinal);

                if (twoDates)
                {
                    alternate.Start = d2;
                    alternate.End   = d2.AddDays(1);
                    alternate.Timex = dateTimex;
                    hasAlternate    = true;
                }

                return true;
            }

            return false;
        }

        private bool ComputeRelativePeriod(int nodeIndex, ref Period period)
        {
            ref var n = ref At(nodeIndex);

            var unit  = n.PeriodUnit;
            int count = Math.Max(1, n.PeriodCount);
            var rel   = n.Relative;
            var today = _reference.Date;

            bool single = count == 1 && rel != RelativeKind.None;

            if (single)
            {
                // "the week after next" is one period beyond the one the qualifier names
                int shift = WeekShift(rel) * (1 + n.ExtraPeriods);

                switch (unit)
                {
                    case TimeUnit.Week:
                    case TimeUnit.WorkWeek:
                    {
                        var monday = StartOfIsoWeek(today).AddDays(shift * 7);
                        period.Start = monday;
                        period.End   = unit == TimeUnit.WorkWeek ? monday.AddDays(5) : monday.AddDays(7);
                        period.Timex = n.Mod == ModKind.RefUndef ? "XXXX-WXX" : $"{IsoYear(monday):0000}-W{IsoWeekOfYear(monday):00}";
                        return true;
                    }

                    case TimeUnit.Fortnight:
                    {
                        var monday = StartOfIsoWeek(today).AddDays(shift * 14);
                        period.Start = monday;
                        period.End   = monday.AddDays(14);
                        period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},P2W)";
                        return true;
                    }

                    case TimeUnit.Weekend:
                    {
                        var monday   = StartOfIsoWeek(today).AddDays(shift * 7);
                        var saturday = monday.AddDays(5);
                        period.Start = saturday;
                        period.End   = saturday.AddDays(2);
                        period.Timex = n.Mod == ModKind.RefUndef ? "XXXX-WXX-WE" : $"{IsoYear(monday):0000}-W{IsoWeekOfYear(monday):00}-WE";
                        return true;
                    }

                    case TimeUnit.Month:
                    {
                        var first = new DateTime(today.Year, today.Month, 1).AddMonths(shift);
                        period.Start = first;
                        period.End   = first.AddMonths(1);
                        period.Timex = n.Mod == ModKind.RefUndef ? "XXXX-XX" : $"{first.Year:0000}-{first.Month:00}";
                        return true;
                    }

                    case TimeUnit.Year:
                    {
                        var first = new DateTime(today.Year + shift, 1, 1);
                        period.Start = first;
                        period.End   = first.AddYears(1);
                        period.Timex = n.Mod == ModKind.RefUndef ? "XXXX" : $"{first.Year:0000}";
                        return true;
                    }

                    case TimeUnit.Quarter:
                    {
                        int quarter = (today.Month - 1) / 3 + shift;
                        int year    = today.Year;

                        while (quarter < 0) { quarter += 4; year--; }
                        while (quarter > 3) { quarter -= 4; year++; }

                        period.Start = new DateTime(year, quarter * 3 + 1, 1);
                        period.End   = period.Start.AddMonths(3);
                        period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},P3M)";
                        return true;
                    }

                    case TimeUnit.Decade:
                    {
                        int decade = today.Year / 10 * 10 + shift * 10;
                        period.Start = new DateTime(decade, 1, 1);
                        period.End   = period.Start.AddYears(10);
                        period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},P10Y)";
                        return true;
                    }

                    case TimeUnit.Day:
                    {
                        var d = today.AddDays(shift);
                        period.Start = d;
                        period.End   = d.AddDays(1);
                        period.Timex = FormatDate(d);
                        return true;
                    }
                }
            }

            // A run of quarters or decades is aligned to the calendar boundary, not to the reference day
            if (unit == TimeUnit.Quarter || unit == TimeUnit.Decade || unit == TimeUnit.Century)
            {
                var current = unit switch
                {
                    TimeUnit.Quarter => new DateTime(today.Year, (today.Month - 1) / 3 * 3 + 1, 1),
                    TimeUnit.Decade  => new DateTime(today.Year / 10  * 10,  1, 1),
                    _                => new DateTime(today.Year / 100 * 100, 1, 1),
                };

                DateTime alignedStart, alignedEnd;

                if (rel == RelativeKind.Last || rel == RelativeKind.Previous || rel == RelativeKind.JustPast)
                {
                    alignedEnd   = current;
                    alignedStart = AddUnits(current, unit, -count);
                }
                else if (rel == RelativeKind.Next || rel == RelativeKind.Coming || rel == RelativeKind.Following)
                {
                    alignedStart = AddUnits(current, unit, 1);
                    alignedEnd   = AddUnits(alignedStart, unit, count);
                }
                else
                {
                    alignedStart = current;
                    alignedEnd   = AddUnits(current, unit, count);
                }

                period.Start = alignedStart;
                period.End   = alignedEnd;
                period.Timex = $"({FormatDate(alignedStart)},{FormatDate(alignedEnd)},{DurationTimexOf(unit, count)})";
                return true;
            }

            // A run of N units, anchored at the reference date
            DateTime start;
            DateTime end;

            if (rel == RelativeKind.Last || rel == RelativeKind.Previous || rel == RelativeKind.JustPast)
            {
                end   = today;
                start = SubtractUnits(today, unit, count);
            }
            else if (rel == RelativeKind.Next || rel == RelativeKind.Coming || rel == RelativeKind.Following)
            {
                start = unit == TimeUnit.BusinessDay ? NextBusinessDay(today) : today.AddDays(1);

                end = AddUnits(start, unit, count);
            }
            else
            {
                start = today;
                end   = AddUnits(today, unit, count);
            }

            period.Start = start;
            period.End   = end;
            period.Timex = $"({FormatDate(start)},{FormatDate(end)},{DurationTimexOf(n.BusinessDays ? TimeUnit.BusinessDay : unit, count)})";
            return true;
        }

        /// <summary>
        /// The weekend a holiday makes long: the nearest saturday and sunday, widened to cover the holiday
        /// itself when it falls on the monday before or the thursday or friday after.
        /// </summary>
        private static Period HolidayWeekend(DateTime holiday)
        {
            var saturday = StartOfIsoWeek(holiday).AddDays(5);

            // A monday or tuesday holiday belongs to the weekend that has just gone
            if ((holiday - saturday).TotalDays < -2.5) saturday = saturday.AddDays(-7);

            var start = holiday < saturday          ? holiday          : saturday;
            var end   = holiday >= saturday.AddDays(2) ? holiday.AddDays(1) : saturday.AddDays(2);

            return new Period
            {
                Start = start,
                End   = end,
                Timex = $"{IsoYear(saturday):0000}-W{IsoWeekOfYear(saturday):00}-WE",
            };
        }

        internal static int WeekShift(RelativeKind rel) => rel switch
        {
            RelativeKind.Next or RelativeKind.Coming or RelativeKind.Following =>  1,
            RelativeKind.AfterNext                                             =>  2,
            RelativeKind.Last or RelativeKind.Previous or RelativeKind.JustPast => -1,
            _                                                                  =>  0,
        };

        private static DateTime NextBusinessDay(DateTime d)
        {
            do { d = d.AddDays(1); } while (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday);
            return d;
        }

        /// <summary>
        /// Moves <paramref name="count"/> business days from <paramref name="d"/>, counting the day it starts
        /// on: four business days from a Tuesday cover Tue-Fri and end on the Saturday.
        /// </summary>
        private static DateTime AddBusinessDays(DateTime d, int count)
        {
            int step = count < 0 ? -1 : 1;

            for (int k = 0; k < Math.Abs(count) - 1; k++)
            {
                do { d = d.AddDays(step); } while (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday);
            }

            return d.AddDays(step);
        }

        private static DateTime AddUnits(DateTime d, TimeUnit unit, int count) => unit switch
        {
            TimeUnit.BusinessDay                                   => AddBusinessDays(d, count),
            TimeUnit.Day or TimeUnit.Night                         => d.AddDays(count),
            TimeUnit.Week or TimeUnit.WorkWeek                     => d.AddDays(count * 7),
            TimeUnit.Fortnight                                     => d.AddDays(count * 14),
            TimeUnit.Weekend                                       => d.AddDays(count * 7),
            TimeUnit.Month                                         => d.AddMonths(count),
            TimeUnit.Quarter                                       => d.AddMonths(count * 3),
            TimeUnit.Year                                          => d.AddYears(count),
            TimeUnit.Decade                                        => d.AddYears(count * 10),
            TimeUnit.Century                                       => d.AddYears(count * 100),
            _                                                      => d.AddDays(count),
        };

        private static DateTime SubtractUnits(DateTime d, TimeUnit unit, int count) => AddUnits(d, unit, -count);

        private static DurationParts Negate(DurationParts p)
        {
            p.Years    = -p.Years;
            p.Months   = -p.Months;
            p.Weeks    = -p.Weeks;
            p.Days     = -p.Days;
            p.Weekends = -p.Weekends;
            p.Hours    = -p.Hours;
            p.Minutes  = -p.Minutes;
            p.Seconds  = -p.Seconds;
            return p;
        }

        private static DateTime AddParts(DateTime d, DurationParts parts)
        {
            return d.AddYears((int)parts.Years)
                    .AddMonths((int)parts.Months)
                    .AddDays(parts.Weeks * 7 + parts.Days + parts.Weekends * 2)
                    .AddHours(parts.Hours)
                    .AddMinutes(parts.Minutes)
                    .AddSeconds(parts.Seconds);
        }

        /// <summary>
        /// An endpoint of a range is reported as a point in time, so its TIMEX is always date-shaped:
        /// "2014-01-01" for a year, "XXXX-03-01" for a month whose year the text never said.
        /// </summary>
        private bool EndpointOf(int nodeIndex, out DateTime start, out string timex, out bool yearUnknown)
        {
            ref var n = ref At(nodeIndex);

            yearUnknown = false;

            if (n.Kind == NodeKind.Date)
            {
                if (ComputeDate(nodeIndex, out timex, out start, out var second, out bool hasSecond))
                {
                    if (timex is object && timex.StartsWith("XXXX-", StringComparison.Ordinal)) yearUnknown = true;

                    return true;
                }

                start = default;
                return false;
            }

            if (ComputePeriod(nodeIndex, out var period, out var alternate, out bool hasAlternate))
            {
                start = period.Start;
                timex = period.Timex;

                if (timex is object && timex.StartsWith("XXXX", StringComparison.Ordinal))
                {
                    yearUnknown = true;
                    if (hasAlternate && alternate.Start.Year == _reference.Year) start = alternate.Start;
                }

                // "from the end of march to the middle of september" — an endpoint is the boundary it names
                if (n.Mod is ModKind.Start or ModKind.Early or ModKind.End or ModKind.Mid or ModKind.Late)
                {
                    var from = start == period.Start ? period : alternate;
                    start    = BoundaryPoint(from.Start, from.End, n.Mod, halfForMid: true);
                }

                // A period endpoint is reported by its first day
                timex = yearUnknown ? $"XXXX-{start.Month:00}-{start.Day:00}" : FormatDate(start);
                return true;
            }

            start       = default;
            timex       = null;
            return false;
        }

        /// <summary>Whether the endpoint is the reference moment rather than a date of its own.</summary>
        private bool IsNow(int nodeIndex)
        {
            ref var n = ref At(nodeIndex);
            return n.PresentRef || (n.Relative == RelativeKind.Current && n.Year < 0 && n.Month < 0 && n.Day < 0
                                                                       && n.Weekday < 0 && n.Holiday == HolidayKind.None
                                                                       && n.OffsetDays == 0 && n.OffsetWeeks == 0
                                                                       && n.OffsetMonths == 0 && n.OffsetYears == 0);
        }

        private bool PrefersMonths(int nodeIndex)
        {
            ref var n = ref At(nodeIndex);
            return n.Quarter > 0 || n.HalfOfYear > 0 || (n.Month >= 0 && n.Day < 0);
        }

        private static string SpanTimex(DateTime from, DateTime to, bool preferMonths = false, bool daysOnly = false)
        {
            if (to < from) (from, to) = (to, from);

            if (daysOnly) return $"P{(int)(to - from).TotalDays}D";

            if (preferMonths)
            {
                int wholeMonths = (to.Year - from.Year) * 12 + (to.Month - from.Month);

                if (wholeMonths > 0) return $"P{wholeMonths}M";
            }

            if (from.Day == to.Day && from.Month == to.Month && to.Year > from.Year) return $"P{to.Year - from.Year}Y";

            int days = (int)(to - from).TotalDays;

            if (days > 0 && days % 7 == 0 && days >= 7 && from.DayOfWeek == to.DayOfWeek && days % 7 == 0 && days <= 28) return $"P{days / 7}W";

            return $"P{days}D";
        }

        private void ResolveDateRange(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            var n = At(nodeIndex);

            if (!ComputePeriod(nodeIndex, out var period, out var alternate, out bool hasAlternate)) return;

            Emit(period);
            if (hasAlternate) Emit(alternate);

            void Emit(Period p)
            {
                var value = new DateTimeResolutionValue { Timex = p.Timex, Type = "daterange", Mod = CombinedModName(n.Mod, n.InnerMod) };

                if (p.NoBounds) value.Value = "not resolved";

                if (!p.NoBounds)
                {
                    var start = p.Start;
                    var end   = p.End;

                    // "the year to date" runs only as far as today
                    if (n.EndsAtReference && _reference.Date < end && _reference.Date > start) end = _reference.Date;

                    if (n.InnerMod != ModKind.None)
                    {
                        // "before the end of december" names the boundary itself, not the last part of it
                        if (IsPointMod(n.Mod))
                        {
                            var point = BoundaryPoint(start, end, n.InnerMod, halfForMid: false, forward: n.Mod is ModKind.After or ModKind.Since);
                            start = point;
                            end   = point;
                        }
                        else
                        {
                            ApplyMod(n.InnerMod, ref start, ref end, out _, out _, thirds: n.Year >= 0);
                        }
                    }

                    ApplyMod(n.Mod, ref start, ref end, out bool dropStart, out bool dropEnd, thirds: n.Year >= 0 && n.InnerMod == ModKind.None);

                    if (!dropStart) value.Start = FormatDate(start);
                    if (!dropEnd)   value.End   = FormatDate(end);
                }

                values.Add(value);
            }
        }

        /// <summary>Narrows or opens a period according to its modifier ("end of", "before", "since", "mid").</summary>
        internal void ApplyMod(ModKind mod, ref DateTime start, ref DateTime end, out bool dropStart, out bool dropEnd) =>
            ApplyMod(mod, ref start, ref end, out dropStart, out dropEnd, thirds: false);

        internal void ApplyMod(ModKind mod, ref DateTime start, ref DateTime end, out bool dropStart, out bool dropEnd, bool thirds)
        {
            dropStart = false;
            dropEnd   = false;

            switch (mod)
            {
                case ModKind.Before:
                    end       = start;
                    dropStart = true;
                    break;

                case ModKind.Until:
                    // "as late as tomorrow" ends on that day, not at the start of the next
                    if ((end - start).TotalDays <= 1) end = start;
                    dropStart = true;
                    break;

                case ModKind.After:
                    // "after january 1, 2007" starts on that day; "after 2010" starts once 2010 is over
                    if ((end - start).TotalDays > 1) start = end;
                    dropEnd = true;
                    break;

                case ModKind.Since:
                    dropEnd = true;
                    break;

                case ModKind.Earlier:
                    // "earlier this month" ends at the halfway point or at the reference, whichever comes first
                    end = Nearer(start, end, first: true, reference: _reference.Date);
                    break;

                case ModKind.Start:
                case ModKind.Early:
                    (start, end) = Slice(start, end, 0);
                    break;

                case ModKind.Mid:
                    (start, end) = Slice(start, end, 1);
                    break;

                case ModKind.Later:
                    // ... and "later this month" starts at whichever of the two comes last
                    start = Nearer(start, end, first: false, reference: _reference.Date);
                    break;

                case ModKind.End:
                    if (thirds) { (start, end) = Slice(start, end, 2); break; }
                    goto case ModKind.Late;

                case ModKind.Late:
                    start = Nearer(start, end, first: false, reference: start);   // the halfway point
                    break;
            }
        }

        /// <summary>
        /// The cut "earlier"/"later" make in a period: the halfway point, moved to the reference moment when
        /// that falls on the inside of it. "earlier this year" stops at today; "later this year" still starts
        /// at midyear, because today is before it.
        /// </summary>
        private static DateTime Nearer(DateTime start, DateTime end, bool first, DateTime reference)
        {
            var span = end - start;
            var half = span.TotalDays >= 300 ? start.AddMonths(6) : start.AddDays((int)(span.TotalDays / 2));

            if (reference <= start || reference >= end) return half;

            return first ? (reference < half ? reference : half)
                         : (reference > half ? reference : half);
        }

        /// <summary>"before the end of december" reports both halves of what it says: "before-end".</summary>
        internal static string CombinedModName(ModKind outer, ModKind inner)
        {
            string name = ModName(outer);

            if (name is null || !IsPointMod(outer) || inner == ModKind.None) return name;

            string part = ModName(inner);
            return part is null ? name : $"{name}-{part}";
        }

        /// <summary>Whether the modifier names a part of the period it sits on ("the end of 2008").</summary>
        private static bool NamesPartOfPeriod(ModKind mod) =>
            mod is ModKind.Start or ModKind.Mid or ModKind.End or ModKind.Early or ModKind.Late;

        private static bool IsPointMod(ModKind mod) =>
            mod is ModKind.Before or ModKind.Until or ModKind.After or ModKind.Since;

        /// <summary>
        /// Where inside a period a modifier points. "the end of december" is the first of january, "the
        /// beginning of march" the first of march, and "mid may" the day the middle of the month gives way.
        /// </summary>
        private static DateTime BoundaryPoint(DateTime start, DateTime end, ModKind mod, bool halfForMid, bool forward = false) => mod switch
        {
            ModKind.Start or ModKind.Early => start,
            ModKind.End                    => forward ? Slice(start, end, 2).Item1 : end,
            ModKind.Mid                    => halfForMid ? start.AddDays((int)((end - start).TotalDays / 2)) : Slice(start, end, 1).Item2,
            ModKind.Late                   => Nearer(start, end, first: false, reference: start),
            _                              => start,
        };

        /// <summary>
        /// The nth week, month or day of a period: "the third month of 2021", "the last week of july",
        /// "the first 2 weeks of 2021".
        /// </summary>
        private Period NthOf(Period host, in Node n, bool yearUnknown)
        {
            var unit   = n.PeriodUnit;
            int count  = Math.Max(1, n.PeriodCount);
            bool weeks = unit == TimeUnit.Week || unit == TimeUnit.WorkWeek;
            var result = new Period();

            if (n.OrdinalFromEnd)
            {
                var end = host.End;

                // ISO names the first week of a year as the one containing its 4th day; the last week is
                // the mirror of that — the one containing the 4th day from the end.
                if (weeks) end = StartOfIsoWeek(end.AddDays(-4)).AddDays(7);

                result.End   = end;
                result.Start = SubtractUnits(end, unit, count);
            }
            else
            {
                var start = host.Start;

                // A year's first week is the ISO one, which can begin in december; a month's is the week
                // its first day falls in
                if (weeks) start = StartOfIsoWeek(host.Start.AddDays(3));

                result.Start = AddUnits(start, unit, n.OrdinalInPeriod - 1);
                result.End   = AddUnits(result.Start, unit, count);
            }

            string yearPart = yearUnknown ? "XXXX" : $"{host.Start.Year:0000}";

            if (weeks && count == 1 && IsOneMonth(host))
            {
                int weekInMonth = n.OrdinalFromEnd ? (host.End.AddDays(-1).Day - 1) / 7 + 1 : n.OrdinalInPeriod;
                result.Timex    = $"{yearPart}-{host.Start.Month:00}-W{weekInMonth:00}";
            }
            else if (weeks && count == 1)
            {
                result.Timex = $"{IsoYear(result.Start):0000}-W{IsoWeekOfYear(result.Start):00}";
            }
            else if (unit == TimeUnit.Month && count == 1)
            {
                result.Timex = $"{yearPart}-{result.Start.Month:00}";
            }
            else
            {
                result.Timex = $"({FormatDate(result.Start)},{FormatDate(result.End)},{DurationTimexOf(unit, count)})";
            }

            return result;
        }

        private static bool IsOneMonth(Period p) => p.End == p.Start.AddMonths(1);

        /// <summary>Splits a period into its early / middle / late thirds, in the shapes a calendar actually uses.</summary>
        private static (DateTime, DateTime) Slice(DateTime start, DateTime end, int which)
        {
            var span = end - start;

            if (span.TotalDays >= 300)   // a year
            {
                return which switch
                {
                    0 => (start, start.AddMonths(4)),
                    1 => (start.AddMonths(4), start.AddMonths(8)),
                    _ => (start.AddMonths(8), end),
                };
            }

            if (span.TotalDays >= 27)    // a month
            {
                return which switch
                {
                    0 => (start, start.AddDays(10)),
                    1 => (start.AddDays(9), start.AddDays(20)),
                    _ => (start.AddDays(20), end),
                };
            }

            if (span.TotalDays >= 6)     // a week
            {
                return which switch
                {
                    0 => (start, start.AddDays(3)),
                    1 => (start.AddDays(2), start.AddDays(5)),
                    _ => (start.AddDays(4), end),
                };
            }

            return (start, end);
        }
    }
}
