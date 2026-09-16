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

            // ---- explicit "A to B"
            if (n.Left >= 0 && n.Right >= 0)
            {
                if (!EndpointOf(n.Left,  out var leftStart,  out var leftTimex,  out bool leftYearUnknown))  return false;
                if (!EndpointOf(n.Right, out var rightStart, out var rightTimex, out bool rightYearUnknown)) return false;

                // A year written on one side applies to both: "nov-feb 2017" starts in 2016
                if (leftYearUnknown && !rightYearUnknown)
                {
                    leftStart = SafeDate(rightStart.Year, leftStart.Month, leftStart.Day);
                    if (leftStart > rightStart) leftStart = leftStart.AddYears(-1);
                    leftYearUnknown = false;
                }
                else if (rightYearUnknown && !leftYearUnknown)
                {
                    rightStart = SafeDate(leftStart.Year, rightStart.Month, rightStart.Day);
                    if (rightStart < leftStart) rightStart = rightStart.AddYears(1);
                    rightYearUnknown = false;
                }
                else if (rightStart < leftStart && leftYearUnknown && rightYearUnknown)
                {
                    rightStart = rightStart.AddYears(1);
                }

                string span = SpanTimex(leftStart, rightStart, PrefersMonths(n.Left) || PrefersMonths(n.Right));

                period.Start = leftStart;
                period.End   = rightStart;
                period.Timex = $"({leftTimex},{rightTimex},{span})";

                if (leftYearUnknown && rightYearUnknown)
                {
                    alternate.Start = leftStart.AddYears(1);
                    alternate.End   = rightStart.AddYears(1);
                    alternate.Timex = period.Timex;
                    hasAlternate    = true;
                }

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

                if (n.Anchor >= 0 && ComputeDate(n.Anchor, out _, out var anchorDate, out _, out _)) start = anchorDate;

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

                var unit  = n.PeriodUnit;
                int count = Math.Max(1, n.PeriodCount);

                if (n.OrdinalFromEnd)
                {
                    var end   = host.End;
                    var start = SubtractUnits(end, unit, count);
                    period.Start = start;
                    period.End   = end;
                }
                else
                {
                    var start = host.Start;

                    if (unit == TimeUnit.Week || unit == TimeUnit.WorkWeek)
                    {
                        start = StartOfIsoWeek(host.Start);
                        if (start < host.Start) start = start.AddDays(7);
                    }

                    start = AddUnits(start, unit, n.OrdinalInPeriod - 1);
                    period.Start = start;
                    period.End   = AddUnits(start, unit, count);
                }

                if ((unit == TimeUnit.Week || unit == TimeUnit.WorkWeek) && count == 1)
                {
                    ref var host2 = ref At(n.Left);

                    if (host2.Month >= 0)
                    {
                        int weekInMonth = (period.Start.Day - 1) / 7 + 1;
                        if (n.OrdinalFromEnd) weekInMonth = (host.End.AddDays(-1).Day - 1) / 7 + 1;

                        string yearPart = host2.Year >= 0 ? $"{host2.Year:0000}" : "XXXX";
                        period.Timex = $"{yearPart}-{host2.Month:00}-W{weekInMonth:00}";
                    }
                    else
                    {
                        period.Timex = $"{IsoYear(period.Start):0000}-W{IsoWeekOfYear(period.Start):00}";
                    }
                }
                else
                {
                    period.Timex = $"({FormatDate(period.Start)},{FormatDate(period.End)},{DurationTimexOf(unit, count)})";
                }

                if (hostHasAlternate)
                {
                    alternate       = period;
                    alternate.Start = period.Start.AddYears(1);
                    alternate.End   = period.End.AddYears(1);
                    hasAlternate    = true;
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

                period.Timex    = n.FiscalKind == 1 ? $"FY{year:0000}" : $"SY{year:0000}";
                period.NoBounds = true;
                return true;
            }

            // ---- a century
            if (n.Century > 0)
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
            if (n.Month >= 0)
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

            bool single = count == 1 && (rel == RelativeKind.This || rel == RelativeKind.Current || rel == RelativeKind.Next || rel == RelativeKind.Last || rel == RelativeKind.Previous || rel == RelativeKind.Coming || rel == RelativeKind.Following);

            if (single)
            {
                int shift = WeekShift(rel);

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

                    case TimeUnit.Weekend:
                    {
                        var monday   = StartOfIsoWeek(today).AddDays(shift * 7);
                        var saturday = monday.AddDays(5);
                        period.Start = saturday;
                        period.End   = saturday.AddDays(2);
                        period.Timex = $"{IsoYear(monday):0000}-W{IsoWeekOfYear(monday):00}-WE";
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

            // A run of N units, anchored at the reference date
            DateTime start;
            DateTime end;

            if (rel == RelativeKind.Last || rel == RelativeKind.Previous)
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

        internal static int WeekShift(RelativeKind rel) => rel switch
        {
            RelativeKind.Next or RelativeKind.Coming or RelativeKind.Following =>  1,
            RelativeKind.Last or RelativeKind.Previous                         => -1,
            _                                                                  =>  0,
        };

        private static DateTime NextBusinessDay(DateTime d)
        {
            do { d = d.AddDays(1); } while (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday);
            return d;
        }

        private static DateTime AddBusinessDays(DateTime d, int count)
        {
            int step = count < 0 ? -1 : 1;

            for (int k = 0; k < Math.Abs(count); k++)
            {
                do { d = d.AddDays(step); } while (d.DayOfWeek == DayOfWeek.Saturday || d.DayOfWeek == DayOfWeek.Sunday);
            }

            return d;
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
                    if (timex is object && timex.StartsWith("XXXX-", StringComparison.Ordinal))
                    {
                        yearUnknown = true;

                        // Inside a range both ends are read in the reference year
                        if (hasSecond) start = start.Year == _reference.Year ? start : second;
                        if (start.Year != _reference.Year && timex.Length == 10) start = SafeDate(_reference.Year, start.Month, start.Day);
                    }

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

                // A period endpoint is reported by its first day
                timex = yearUnknown ? $"XXXX-{start.Month:00}-{start.Day:00}" : FormatDate(start);
                return true;
            }

            start       = default;
            timex       = null;
            return false;
        }

        private bool PrefersMonths(int nodeIndex)
        {
            ref var n = ref At(nodeIndex);
            return n.Quarter > 0 || n.HalfOfYear > 0 || (n.Month >= 0 && n.Day < 0);
        }

        private static string SpanTimex(DateTime from, DateTime to, bool preferMonths = false)
        {
            if (to < from) (from, to) = (to, from);

            if (from.Day == to.Day)
            {
                int wholeMonths = (to.Year - from.Year) * 12 + (to.Month - from.Month);

                if (preferMonths && wholeMonths > 0) return $"P{wholeMonths}M";
            }

            if (from.Day == to.Day && from.Month == to.Month && to.Year > from.Year) return $"P{to.Year - from.Year}Y";

            if (from.Day == to.Day)
            {
                int months = (to.Year - from.Year) * 12 + (to.Month - from.Month);
                if (months > 0) return $"P{months}M";
            }

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
                var value = new DateTimeResolutionValue { Timex = p.Timex, Type = "daterange", Mod = ModName(n.Mod) };

                if (p.NoBounds) value.Value = "not resolved";

                if (!p.NoBounds)
                {
                    var start = p.Start;
                    var end   = p.End;

                    ApplyMod(n.Mod, ref start, ref end, out bool dropStart, out bool dropEnd);

                    if (!dropStart) value.Start = FormatDate(start);
                    if (!dropEnd)   value.End   = FormatDate(end);
                }

                values.Add(value);
            }
        }

        /// <summary>Narrows or opens a period according to its modifier ("end of", "before", "since", "mid").</summary>
        internal static void ApplyMod(ModKind mod, ref DateTime start, ref DateTime end, out bool dropStart, out bool dropEnd)
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
                    dropStart = true;
                    break;

                case ModKind.After:
                    start   = end;
                    dropEnd = true;
                    break;

                case ModKind.Since:
                    dropEnd = true;
                    break;

                case ModKind.Start:
                case ModKind.Early:
                case ModKind.Earlier:
                    (start, end) = Half(start, end, first: true);
                    if (mod != ModKind.Earlier) (start, end) = Slice(start, end, 0);
                    break;

                case ModKind.Mid:
                    (start, end) = Slice(start, end, 1);
                    break;

                case ModKind.End:
                case ModKind.Late:
                case ModKind.Later:
                    if (mod == ModKind.Later) { (start, end) = Half(start, end, first: false); }
                    else                      { (start, end) = Slice(start, end, 2); }
                    break;
            }
        }

        /// <summary>Keeps the first or the last half of a period, which is what "earlier"/"later this month" mean.</summary>
        private static (DateTime, DateTime) Half(DateTime start, DateTime end, bool first)
        {
            var span = end - start;

            if (span.TotalDays >= 300) return first ? (start, start.AddMonths(6)) : (start.AddMonths(6), end);

            int days = (int)(span.TotalDays / 2);

            return first ? (start, start.AddDays(days)) : (start.AddDays(days), end);
        }

        /// <summary>Splits a period into its early / middle / late thirds, in the shapes a calendar actually uses.</summary>
        private static (DateTime, DateTime) Slice(DateTime start, DateTime end, int which)
        {
            var span = end - start;

            if (span.TotalDays >= 300)   // a year
            {
                return which switch
                {
                    0 => (start, start.AddMonths(6)),
                    1 => (start.AddMonths(4), start.AddMonths(8)),
                    _ => (start.AddMonths(6), end),
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
