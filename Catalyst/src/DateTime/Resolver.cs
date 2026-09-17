using System;
using System.Collections.Generic;
using System.Globalization;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>
    /// Turns a parsed <see cref="Node"/> into concrete dates and TIMEX strings, measured against a reference moment.
    /// </summary>
    public sealed partial class Resolver
    {
        private readonly Node[]   _nodes;
        private readonly DateTime _reference;
        private readonly Lexicon  _lexicon;

        public Resolver(Node[] nodes, DateTime reference, Lexicon lexicon)
        {
            _nodes     = nodes;
            _reference = reference;
            _lexicon   = lexicon;
        }

        private ref Node At(int index) => ref _nodes[index];

        // ------------------------------------------------------------------ formatting

        /// <summary>What a date whose calendar day does not exist resolves to, matching the reference implementation.</summary>
        internal const string NotResolved = "not resolved";

        internal static string FormatDate(DateTime d)     => d.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        internal static string FormatTime(DateTime d)     => d.ToString("HH:mm:ss",   CultureInfo.InvariantCulture);
        internal static string FormatDateTime(DateTime d) => d.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

        internal static string TimexOfTime(int hour, int minute, int second)
        {
            if (second >= 0) return $"T{hour:00}:{minute:00}:{second:00}";
            if (minute >= 0) return $"T{hour:00}:{minute:00}";
            return $"T{hour:00}";
        }

        internal static DateTime StartOfIsoWeek(DateTime d)
        {
            int delta = ((int)d.DayOfWeek + 6) % 7;   // Monday = 0
            return d.Date.AddDays(-delta);
        }

        internal static int IsoWeekOfYear(DateTime d) => ISOWeek.GetWeekOfYear(d.Date);

        internal static int IsoYear(DateTime d) => ISOWeek.GetYear(d.Date);

        /// <summary>The TIMEX day-of-week digit: Monday is 1, Sunday is 7.</summary>
        internal static int TimexWeekday(int sundayBased) => sundayBased == 0 ? 7 : sundayBased;

        // ------------------------------------------------------------------ entry point

        public DateTimeEntity Resolve(int nodeIndex, string text)
        {
            ref var n = ref At(nodeIndex);

            var values = new List<DateTimeResolutionValue>(2);

            switch (n.Kind)
            {
                case NodeKind.Date:          ResolveDate(nodeIndex, values);          break;
                case NodeKind.Time:          ResolveTime(nodeIndex, values);          break;
                case NodeKind.DateTime:      ResolveDateTime(nodeIndex, values);      break;
                case NodeKind.DateRange:     ResolveDateRange(nodeIndex, values);     break;
                case NodeKind.TimeRange:     ResolveTimeRange(nodeIndex, values);     break;
                case NodeKind.DateTimeRange: ResolveDateTimeRange(nodeIndex, values); break;
                case NodeKind.Duration:      ResolveDuration(nodeIndex, values);      break;
                case NodeKind.Set:           ResolveSet(nodeIndex, values);           break;
            }

            if (values.Count == 0) return null;

            var kind = n.Kind switch
            {
                NodeKind.Date          => DateTimeEntityKind.Date,
                NodeKind.Time          => DateTimeEntityKind.Time,
                NodeKind.DateTime      => DateTimeEntityKind.DateTime,
                NodeKind.DateRange     => DateTimeEntityKind.DateRange,
                NodeKind.TimeRange     => DateTimeEntityKind.TimeRange,
                NodeKind.DateTimeRange => DateTimeEntityKind.DateTimeRange,
                NodeKind.Duration      => DateTimeEntityKind.Duration,
                _                      => DateTimeEntityKind.Set,
            };

            return new DateTimeEntity
            {
                Text   = text,
                Kind   = kind,
                Values = values,
            };
        }

        internal static string ModName(ModKind mod) => mod switch
        {
            ModKind.Before  => "before",
            ModKind.After   => "after",
            ModKind.Since   => "since",
            ModKind.Until   => "until",
            ModKind.Start   => "start",
            ModKind.End     => "end",
            ModKind.Mid     => "mid",
            ModKind.Approx  => "approx",
            ModKind.RefUndef=> "ref_undef",
            ModKind.Less    => "less",
            ModKind.More    => "more",
            ModKind.Early   => "start",
            ModKind.Late    => "end",
            ModKind.Earlier => null,
            ModKind.Later   => null,
            _               => null,
        };

        // ------------------------------------------------------------------ dates

        /// <summary>The concrete dates a date node can mean, together with the TIMEX that describes it.</summary>
        internal bool ComputeDate(int nodeIndex, out string timex, out DateTime first, out DateTime second, out bool hasSecond)
        {
            ref var n = ref At(nodeIndex);

            timex     = null;
            first     = default;
            second    = default;
            hasSecond = false;

            // --- a holiday
            if (n.Holiday != HolidayKind.None)
            {
                int holidayYear = n.Year >= 0 ? n.Year : _reference.Year;
                var d           = Holidays.Resolve(n.Holiday, holidayYear);

                if (n.Year < 0)
                {
                    switch (n.Relative)
                    {
                        case RelativeKind.Next:
                        case RelativeKind.Coming:
                        case RelativeKind.Following:
                            if (d <= _reference.Date) d = Holidays.Resolve(n.Holiday, holidayYear + 1);
                            break;

                        case RelativeKind.Last:
                        case RelativeKind.Previous:
                            if (d >= _reference.Date) d = Holidays.Resolve(n.Holiday, holidayYear - 1);
                            break;
                    }
                }

                if (n.Year < 0 && n.Relative == RelativeKind.None)
                {
                    // "before independence day" — the same two nearest readings a year-less date has
                    if (d >= _reference.Date) { first = Holidays.Resolve(n.Holiday, holidayYear - 1); second = d; }
                    else                      { first = d; second = Holidays.Resolve(n.Holiday, holidayYear + 1); }

                    // A holiday on the same day every year is named by that day; one that moves has no
                    // day to name, so the timex says only that the year is open
                    timex     = Holidays.IsFixedDate(n.Holiday) || _lexicon.MovableHolidayNamesItsDay
                              ? $"XXXX-{first.Month:00}-{first.Day:00}"
                              : "XXXX";
                    hasSecond = true;
                    return true;
                }

                timex = FormatDate(d);
                first = d;
                return true;
            }

            // --- an offset from an anchor
            if (n.OffsetYears != 0 || n.OffsetMonths != 0 || n.OffsetWeeks != 0 || n.OffsetDays != 0 || n.Anchor >= 0 || (n.Relative == RelativeKind.Current && n.Weekday < 0 && n.Year < 0 && n.Month < 0 && n.Day < 0))
            {
                DateTime anchor = _reference.Date;

                if (n.Anchor >= 0 && ComputeAnchorDate(n.Anchor, out var anchorDate))
                {
                    anchor = anchorDate;
                }

                var d = anchor.AddYears(n.OffsetYears).AddMonths(n.OffsetMonths).AddDays(n.OffsetWeeks * 7 + n.OffsetDays);

                // "the 15th day of next month" — the offset moves the month, the day is still named
                if (n.Day >= 0 && n.Month < 0 && n.OffsetDays == 0 && n.OffsetWeeks == 0) d = SafeDate(d.Year, d.Month, n.Day);

                if (n.Weekday >= 0)
                {
                    d = WeekdayInWeekOf(d, n.Weekday);
                }

                timex = FormatDate(d);
                first = d;
                return true;
            }

            // --- a weekday
            if (n.Weekday >= 0 && n.Year < 0 && n.Month < 0 && n.Day < 0)
            {
                var d = WeekdayInWeekOf(_reference.Date, n.Weekday);

                switch (n.Relative)
                {
                    case RelativeKind.Next:
                    case RelativeKind.Following:
                        d = d.AddDays(7);
                        break;

                    case RelativeKind.AfterNext:
                        d = d.AddDays(14);
                        break;

                    case RelativeKind.BeforeLast:
                        d = d.AddDays(-14);
                        break;

                    case RelativeKind.Coming:
                        if (d <= _reference.Date) d = d.AddDays(7);
                        break;

                    case RelativeKind.JustPast:
                        // "past wednesday" is the most recent one, which may still be in this week
                        if (d >= _reference.Date) d = d.AddDays(-7);
                        break;

                    case RelativeKind.Last:
                    case RelativeKind.Previous:
                        d = d.AddDays(-7);
                        break;
                }

                if (n.Relative == RelativeKind.None)
                {
                    timex = $"XXXX-WXX-{TimexWeekday(n.Weekday)}";

                    {
                        if (d >= _reference.Date) { first = d.AddDays(-7); second = d; }
                        else                      { first = d;             second = d.AddDays(7); }

                        hasSecond = true;
                        return true;
                    }
                }

                timex = FormatDate(d);

                first = d;
                return true;
            }

            // --- calendar fields
            int  day   = n.Day;
            int  month = n.Month;
            int  year  = n.Year;

            if (day < 0 && month < 0 && year < 0) return false;

            if (day < 0) return false;   // a bare month or year is a period, not a date

            if (month < 0)
            {
                month = _reference.Month;
                timex = $"XXXX-XX-{day:00}";

                var thisMonth = SafeDate(_reference.Year, month, day);

                if (n.Relative == RelativeKind.Next)
                {
                    first = thisMonth.AddMonths(1);
                    timex = FormatDate(first);
                    return true;
                }

                if (n.Relative == RelativeKind.Last)
                {
                    first = thisMonth.AddMonths(-1);
                    timex = FormatDate(first);
                    return true;
                }

                // "around the 21st this month" names one day, not the two nearest
                if (n.Relative == RelativeKind.This || n.Relative == RelativeKind.Current)
                {
                    first = thisMonth;
                    timex = FormatDate(first);
                    return true;
                }

                // "monday 21" — the weekday picks which months can be meant: the nearest one behind and the nearest ahead
                if (n.Weekday >= 0)
                {
                    timex = $"XXXX-WXX-{TimexWeekday(n.Weekday)}";

                    var  anchor   = new DateTime(_reference.Year, _reference.Month, 1);
                    bool haveBack = false;
                    bool haveNext = false;

                    for (int step = 0; step <= 120 && (!haveBack || !haveNext); step++)
                    {
                        if (!haveNext && TryExactDate(anchor.AddMonths(step), day, out var ahead)
                            && ahead >= _reference.Date && (int)ahead.DayOfWeek == n.Weekday)
                        {
                            second   = ahead;
                            haveNext = true;
                        }

                        if (!haveBack && TryExactDate(anchor.AddMonths(-step), day, out var behind)
                            && behind < _reference.Date && (int)behind.DayOfWeek == n.Weekday)
                        {
                            first    = behind;
                            haveBack = true;
                        }
                    }

                    // "monday the 26th" names one day — the nearer of the two; "monday 21" leaves both open
                    if (n.DefiniteDay && (haveNext || haveBack))
                    {
                        if (haveNext && (!haveBack || second - _reference.Date <= _reference.Date - first)) first = second;

                        timex = FormatDate(first);
                        return true;
                    }

                    if (haveBack && haveNext) { hasSecond = true; return true; }
                    if (haveNext)             { first = second;   return true; }
                    if (haveBack)             { return true; }

                    first = thisMonth;
                    return true;
                }

                if (thisMonth >= _reference.Date)
                {
                    first  = thisMonth.AddMonths(-1);
                    second = thisMonth;
                }
                else
                {
                    first  = thisMonth;
                    second = thisMonth.AddMonths(1);
                }

                hasSecond = true;
                return true;
            }

            if (year < 0)
            {
                timex = $"XXXX-{month:00}-{day:00}";

                // "feb 30" is a day no year has
                if (!IsPossibleDayOfMonth(month, day)) { first = default; return true; }

                // "next 6th of april" names one year, and so one day
                if (n.Relative != RelativeKind.None)
                {
                    int shift = n.Relative switch
                    {
                        RelativeKind.Next or RelativeKind.Coming or RelativeKind.Following  =>  1,
                        RelativeKind.AfterNext                                              =>  2,
                        RelativeKind.Last or RelativeKind.Previous or RelativeKind.JustPast => -1,
                        RelativeKind.BeforeLast                                              => -2,
                        _                                                                   =>  0,
                    };

                    first = SafeDate(_reference.Year + shift, month, day);
                    return true;
                }

                bool haveBack = false;
                bool haveNext = false;

                // "feb 29" skips the years that do not have the day; today itself is the reading behind
                for (int step = 0; step <= 8 && (!haveBack || !haveNext); step++)
                {
                    if (!haveNext && TryExactDate(new DateTime(_reference.Year + step, month, 1), day, out var ahead)
                        && ahead > _reference.Date)
                    {
                        second   = ahead;
                        haveNext = true;
                    }

                    if (!haveBack && TryExactDate(new DateTime(_reference.Year - step, month, 1), day, out var behind)
                        && behind <= _reference.Date)
                    {
                        first    = behind;
                        haveBack = true;
                    }
                }

                if (haveBack && haveNext) { hasSecond = true; return true; }
                if (haveNext)             { first = second;   return true; }
                if (haveBack)             { return true; }

                first = default;
                return true;
            }

            timex = $"{year:0000}-{month:00}-{day:00}";
            first = TryExactDate(new DateTime(year, month, 1), day, out var exact) ? exact : default;
            return true;
        }

        /// <summary>Whether any year at all has this day in this month — "feb 30" has none.</summary>
        private static bool IsPossibleDayOfMonth(int month, int day)
        {
            if (month < 1 || month > 12) return false;
            return day >= 1 && day <= (month == 2 ? 29 : DateTime.DaysInMonth(2001, month));
        }

        /// <summary>The day in <paramref name="month"/>, or false when that month is too short for it.</summary>
        private static bool TryExactDate(DateTime month, int day, out DateTime date)
        {
            date = default;
            if (day < 1 || day > DateTime.DaysInMonth(month.Year, month.Month)) return false;
            date = new DateTime(month.Year, month.Month, day);
            return true;
        }

        /// <summary>
        /// The date an offset counts from. "2 weeks before christmas" means the christmas that is coming,
        /// and "3 days from tuesday" the tuesday that is coming — never the one that has gone.
        /// </summary>
        private bool ComputeAnchorDate(int nodeIndex, out DateTime anchor)
        {
            anchor = default;

            if (!ComputeDate(nodeIndex, out _, out var first, out var second, out bool hasSecond)) return false;

            anchor = hasSecond ? second : first;

            if (!hasSecond && anchor < _reference.Date)
            {
                ref var a = ref At(nodeIndex);

                if (a.Holiday != HolidayKind.None && a.Year < 0 && a.Relative == RelativeKind.None)
                {
                    anchor = Holidays.Resolve(a.Holiday, _reference.Year + 1);
                }
            }

            return true;
        }

        /// <summary>
        /// The date a duration is measured from. "for 1 week, from 20th dec" means this year's december,
        /// whether that has gone or is still coming.
        /// </summary>
        private bool ComputeAnchorDateInReferenceYear(int nodeIndex, out DateTime anchor)
        {
            ref var a = ref At(nodeIndex);

            if (a.Year < 0 && a.Month >= 0 && a.Day >= 0 && a.Holiday == HolidayKind.None && a.Anchor < 0)
            {
                anchor = SafeDate(_reference.Year, a.Month, a.Day);
                return true;
            }

            return ComputeAnchorDate(nodeIndex, out anchor);
        }

        internal static DateTime SafeDate(int year, int month, int day)
        {
            if (year  < 1) year  = 1;
            if (year  > 9999) year = 9999;
            if (month < 1) month = 1;
            if (month > 12) month = 12;

            int max = DateTime.DaysInMonth(year, month);
            bool overflow = day > max;
            int clamped = day < 1 ? 1 : (day > max ? max : day);

            var d = new DateTime(year, month, clamped);

            // "feb 30" rolls forward the way a calendar would
            if (overflow) d = d.AddDays(day - max);

            return d;
        }

        private DateTime WeekdayInWeekOf(DateTime reference, int sundayBasedWeekday)
        {
            var monday = StartOfIsoWeek(reference);
            int offset = (sundayBasedWeekday + 6) % 7;   // Monday = 0 .. Sunday = 6
            return monday.AddDays(offset);
        }

        private void ResolveDate(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            ref var n = ref At(nodeIndex);

            if (!ComputeDate(nodeIndex, out var timex, out var first, out var second, out bool hasSecond)) return;

            string mod = ModName(n.Mod);

            // "around the 21st" is reported as the range it approximates
            string type = n.Mod == ModKind.Approx ? "daterange" : "date";

            values.Add(new DateTimeResolutionValue { Timex = timex, Type = type, Value = first == default ? NotResolved : FormatDate(first), Mod = mod });

            if (hasSecond)
            {
                values.Add(new DateTimeResolutionValue { Timex = timex, Type = type, Value = FormatDate(second), Mod = mod });
            }
        }

        // ------------------------------------------------------------------ times

        /// <summary>Resolves a clock reading, returning the one or two hours it can mean.</summary>
        internal void ComputeTime(int nodeIndex, out int firstHour, out int secondHour, out bool hasSecond, out int minute, out int second)
        {
            ref var n = ref At(nodeIndex);

            // "a las 24" is midnight of the day it names, and 24 is not an hour a DateTime can hold
            int hour = n.Hour == 24 ? 0 : n.Hour;
            minute   = n.Minute;
            second   = n.Second;

            int ampm = n.AmPm;

            if (ampm < 0 && n.PartOfDay != PartOfDayKind.None)
            {
                ampm = n.PartOfDay switch
                {
                    PartOfDayKind.Morning or PartOfDayKind.EarlyMorning or PartOfDayKind.Breakfast
                        or PartOfDayKind.ThisMorning                                               => 0,
                    PartOfDayKind.Afternoon or PartOfDayKind.Evening or PartOfDayKind.Night
                        or PartOfDayKind.Tonight or PartOfDayKind.LastNight or PartOfDayKind.ThisEvening or PartOfDayKind.ThisAfternoon
                        or PartOfDayKind.Dinner                                                    => 1,
                    _                                                                              => -1,
                };
            }

            hasSecond  = false;
            secondHour = -1;

            if (ampm == 0)
            {
                firstHour = hour == 12 ? 0 : hour;
            }
            else if (ampm == 1)
            {
                firstHour = hour < 12 ? hour + 12 : hour;
            }
            else
            {
                firstHour = hour;

                if (hour >= 1 && hour <= 11)
                {
                    secondHour = hour + 12;
                    hasSecond  = true;
                }
                else if (hour == 12)
                {
                    secondHour = 0;
                    hasSecond  = true;
                }
            }
        }

        private void ResolveTime(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            ref var n = ref At(nodeIndex);

            ComputeTime(nodeIndex, out int h1, out int h2, out bool hasSecond, out int minute, out int second);

            if (h1 < 0) return;

            int m = minute < 0 ? 0 : minute;
            int s = second < 0 ? 0 : second;

            string mod = ModName(n.Mod);

            // "around 1pm" is reported as the range it approximates
            string type = n.Mod == ModKind.Approx ? "timerange" : "time";

            values.Add(new DateTimeResolutionValue
            {
                Timex = TimexOfTime(h1, minute, second),
                Type  = type,
                Value = $"{h1:00}:{m:00}:{s:00}",
                Mod   = mod,
            });

            if (hasSecond)
            {
                values.Add(new DateTimeResolutionValue
                {
                    Timex = TimexOfTime(h2, minute, second),
                    Type  = type,
                    Value = $"{h2:00}:{m:00}:{s:00}",
                    Mod   = mod,
                });
            }
        }

        // ------------------------------------------------------------------ datetimes

        private void ResolveDateTime(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            ref var n = ref At(nodeIndex);

            // "now" / "in 5 minutes" / "30 min later"
            if (n.Left >= 0 && At(n.Left).Kind == NodeKind.Duration)
            {
                var parts  = At(n.Left).Duration;
                double sec = parts.TotalSeconds * (n.OffsetDays < 0 ? -1 : 1);
                var    when = _reference.AddSeconds(sec);

                values.Add(new DateTimeResolutionValue
                {
                    Timex = FormatDate(when) + TimexOfTime(when.Hour, when.Minute, when.Second),
                    Type  = "datetime",
                    Value = FormatDateTime(when),
                });
                return;
            }

            if (n.Hour < 0 && n.Relative == RelativeKind.Current && n.Year < 0 && n.Month < 0 && n.Day < 0 && n.Weekday < 0 && n.Holiday == HolidayKind.None && n.OffsetDays == 0)
            {
                values.Add(new DateTimeResolutionValue
                {
                    Timex = "PRESENT_REF",
                    Type  = "datetime",
                    Value = FormatDateTime(_reference),
                });
                return;
            }

            if (!ComputeDate(nodeIndex, out var dateTimex, out var d1, out var d2, out bool hasSecondDate))
            {
                d1        = _reference.Date;
                dateTimex = FormatDate(d1);
            }

            ComputeTime(nodeIndex, out int h1, out int h2, out bool hasSecondTime, out int minute, out int second);

            if (h1 < 0) return;

            int m = minute < 0 ? 0 : minute;
            int s = second < 0 ? 0 : second;

            string mod = ModName(n.Mod);

            // "around tomorrow 10am" is reported as the range it approximates
            string type = n.Mod == ModKind.Approx ? "datetimerange" : "datetime";

            void Emit(DateTime date, int hour)
            {
                values.Add(new DateTimeResolutionValue
                {
                    Timex = dateTimex + TimexOfTime(hour, minute, second),
                    Type  = type,
                    Value = FormatDateTime(new DateTime(date.Year, date.Month, date.Day, hour, m, s)),
                    Mod   = mod,
                });
            }

            Emit(d1, h1);
            if (hasSecondDate) Emit(d2, h1);

            if (hasSecondTime)
            {
                Emit(d1, h2);
                if (hasSecondDate) Emit(d2, h2);
            }
        }

        // ------------------------------------------------------------------ durations

        private void ResolveDuration(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            ref var n = ref At(nodeIndex);

            values.Add(new DateTimeResolutionValue
            {
                Timex = n.DurationTimex,
                Type  = "duration",
                Value = DurationParts.Fmt(n.DurationSeconds),
                Mod   = ModName(n.Mod),
            });
        }

        // ------------------------------------------------------------------ sets

        private void ResolveSet(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            ref var n = ref At(nodeIndex);

            string timex;

            string partOfDay = n.PartOfDay != PartOfDayKind.None ? Parser.RangeOf(n.PartOfDay).Timex
                             : n.SetUnit == TimeUnit.Night       ? Parser.RangeOf(PartOfDayKind.Night).Timex
                             : null;

            if (n.Weekday >= 0)
            {
                timex = $"XXXX-WXX-{TimexWeekday(n.Weekday)}";

                if (n.Hour >= 0)
                {
                    ComputeTime(nodeIndex, out int hour, out _, out _, out int minute, out int second);
                    timex += TimexOfTime(hour, minute, second);
                }
                else if (partOfDay is object)
                {
                    timex += partOfDay;
                }
            }
            else if (n.Day >= 0)
            {
                timex = $"XXXX-XX-{n.Day:00}";
            }
            else if (n.Hour >= 0)
            {
                ComputeTime(nodeIndex, out int hour, out _, out _, out int minute, out int second);
                timex = TimexOfTime(hour, minute, second);
            }
            else if (partOfDay is object)
            {
                timex = partOfDay;
            }
            else if (n.BusinessDays)
            {
                timex = $"P{n.SetInterval}WD";
            }
            else
            {
                timex = DurationTimexOf(n.SetUnit, n.SetInterval);
            }

            values.Add(new DateTimeResolutionValue
            {
                Timex = timex,
                Type  = "set",
                Value = "not resolved",
            });
        }

        internal static string DurationTimexOf(TimeUnit unit, int count) => unit switch
        {
            TimeUnit.Second      => $"PT{count}S",
            TimeUnit.Minute      => $"PT{count}M",
            TimeUnit.Hour        => $"PT{count}H",
            TimeUnit.Day         => $"P{count}D",
            TimeUnit.BusinessDay => $"P{count}BD",
            TimeUnit.Night       => $"P{count}D",
            TimeUnit.Week        => $"P{count}W",
            TimeUnit.WorkWeek    => $"P{count}W",
            TimeUnit.HalfYear    => $"P{count * 0.5}Y",
            TimeUnit.Fortnight   => $"P{count * 2}W",
            TimeUnit.Weekend     => $"P{count}WE",
            TimeUnit.Month       => $"P{count}M",
            TimeUnit.Quarter     => $"P{count * 3}M",
            TimeUnit.Year        => $"P{count}Y",
            TimeUnit.Decade      => $"P{count * 10}Y",
            TimeUnit.Century     => $"P{count * 100}Y",
            _                    => $"P{count}D",
        };
    }
}
