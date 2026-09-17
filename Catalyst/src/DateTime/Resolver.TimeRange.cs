using System;
using System.Collections.Generic;

namespace Catalyst.DateTimeRecognition
{
    public sealed partial class Resolver
    {
        /// <summary>A resolved clock interval, as hour/minute pairs plus the TIMEX that names it.</summary>
        internal struct ClockRange
        {
            public int    StartHour, StartMinute, StartSecond;
            public int    EndHour,   EndMinute,   EndSecond;
            public string StartTimex;
            public string EndTimex;
            public string Timex;
            public bool   Open;        // a modifier left one side unbounded
            public bool   EndNextDay;
        }

        internal bool ComputeClockRange(int nodeIndex, out ClockRange range, out ClockRange alternate, out bool hasAlternate)
        {
            ref var n = ref At(nodeIndex);

            range        = default;
            alternate    = default;
            hasAlternate = false;

            // ---- a named part of the day
            if (n.PartOfDay != PartOfDayKind.None && n.Left < 0)
            {
                var part = Parser.RangeOf(n.PartOfDay);
                if (part.Timex is null) return false;

                int startHour = part.StartHour, startMinute = part.StartMinute;
                int endHour   = part.EndHour,   endMinute   = part.EndMinute, endSecond = part.EndSecond;

                if (n.Mod == ModKind.Early || n.Mod == ModKind.Start)
                {
                    endHour   = (part.StartHour + part.EndHour) / 2;
                    endMinute = 0;
                    endSecond = 0;
                }
                else if (n.Mod == ModKind.Late || n.Mod == ModKind.End)
                {
                    startHour   = (part.StartHour + part.EndHour) / 2;
                    startMinute = 0;
                }

                range.StartHour   = startHour;
                range.StartMinute = startMinute;
                range.EndHour     = endHour;
                range.EndMinute   = endMinute;
                range.EndSecond   = endSecond;
                range.Timex       = part.Timex;
                range.StartTimex  = part.Timex;
                range.EndTimex    = part.Timex;
                return true;
            }

            // ---- a day cut into hours: "mid today", "later in today", "early in the day wednesday"
            if (n.PartOfDay == PartOfDayKind.None && n.Left < 0 && n.Hour < 0 && n.Kind == NodeKind.DateTimeRange
                && Parser.IsDaySlice(n.Mod))
            {
                (range.StartHour, range.EndHour) = n.Mod switch
                {
                    ModKind.Start or ModKind.Early or ModKind.Earlier => (0, 12),
                    ModKind.Mid                                       => (10, 14),
                    _                                                 => (12, 24),
                };

                range.Timex      = null;
                range.StartTimex = null;
                range.EndTimex   = null;
                return true;
            }

            // ---- a time plus a duration: "for 2 hours from 2pm"
            if (n.Left >= 0 && n.RangeDuration >= 0 && At(n.RangeDuration).Kind == NodeKind.Duration)
            {
                ComputeTime(n.Left, out int h1, out _, out _, out int m1, out int s1);
                if (h1 < 0) return false;

                var parts = At(n.RangeDuration).Duration;
                var start = new DateTime(2000, 1, 1, h1, m1 < 0 ? 0 : m1, s1 < 0 ? 0 : s1);
                var end   = start.AddHours(parts.Hours).AddMinutes(parts.Minutes).AddSeconds(parts.Seconds);

                range.StartHour   = h1;
                range.StartMinute = m1 < 0 ? 0 : m1;
                range.StartSecond = s1 < 0 ? 0 : s1;
                range.EndHour     = end.Hour;
                range.EndMinute   = end.Minute;
                range.EndSecond   = end.Second;
                range.StartTimex  = TimexOfTime(h1, m1, s1);
                range.EndTimex    = TimexOfTime(end.Hour, end.Minute, end.Second);
                range.Timex       = $"({range.StartTimex},{range.EndTimex},{ClockSpanTimex(range)})";
                return true;
            }

            // ---- an explicit "A to B"
            if (n.Left >= 0 && n.Right >= 0)
            {
                if (!ComputeExplicitClockRange(n.Left, n.Right, out range, out alternate, out hasAlternate)) return false;

                // "this evening from 7 to 9" — the part of the day settles which of the two readings is meant
                if (hasAlternate && n.PartOfDay != PartOfDayKind.None)
                {
                    bool afternoon = n.PartOfDay is PartOfDayKind.Afternoon or PartOfDayKind.Evening
                                                 or PartOfDayKind.Night     or PartOfDayKind.Tonight
                                                 or PartOfDayKind.Dinner;

                    if (afternoon == (alternate.StartHour >= 12) && afternoon != (range.StartHour >= 12))
                    {
                        range = alternate;
                    }

                    hasAlternate = false;
                }

                return true;
            }

            // ---- a single time carrying a modifier ("after 3pm")
            if (n.Left >= 0 || n.Hour >= 0)
            {
                int timeNode = n.Left >= 0 ? n.Left : nodeIndex;

                ComputeTime(timeNode, out int h1, out int h2, out bool twoHours, out int minute, out int second);
                if (h1 < 0) return false;

                range.StartHour   = h1;
                range.StartMinute = minute < 0 ? 0 : minute;
                range.StartSecond = second < 0 ? 0 : second;
                range.EndHour     = h1;
                range.EndMinute   = range.StartMinute;
                range.EndSecond   = range.StartSecond;
                range.Timex       = TimexOfTime(h1, minute, second);
                range.StartTimex  = range.Timex;
                range.EndTimex    = range.Timex;
                range.Open        = true;

                if (twoHours)
                {
                    alternate            = range;
                    alternate.StartHour  = h2;
                    alternate.EndHour    = h2;
                    alternate.Timex      = TimexOfTime(h2, minute, second);
                    alternate.StartTimex = alternate.Timex;
                    alternate.EndTimex   = alternate.Timex;
                    hasAlternate         = true;
                }

                return true;
            }

            return false;
        }

        private bool ComputeExplicitClockRange(int leftNode, int rightNode, out ClockRange range, out ClockRange alternate, out bool hasAlternate)
        {
            range        = default;
            alternate    = default;
            hasAlternate = false;

            ref var left  = ref At(leftNode);
            ref var right = ref At(rightNode);

            // A trailing am/pm or part of the day carries back to the opening time: "5 to 6pm"
            int sharedAmPm = right.AmPm;

            if (sharedAmPm < 0 && right.PartOfDay != PartOfDayKind.None)
            {
                sharedAmPm = right.PartOfDay switch
                {
                    PartOfDayKind.Morning or PartOfDayKind.EarlyMorning => 0,
                    PartOfDayKind.Afternoon or PartOfDayKind.Evening or PartOfDayKind.Night or PartOfDayKind.Tonight => 1,
                    _ => -1,
                };
            }

            ComputeTime(rightNode, out int endHour, out int endHourAlt, out bool endTwo, out int endMinute, out int endSecond);

            int leftHour, leftHourAlt = -1;
            bool leftTwo = false;
            int leftMinute, leftSecond;

            if (left.AmPm < 0 && left.PartOfDay == PartOfDayKind.None && sharedAmPm >= 0 && left.Hour >= 0 && left.Hour <= 12)
            {
                leftHour   = sharedAmPm == 1 ? (left.Hour < 12 ? left.Hour + 12 : left.Hour) : (left.Hour == 12 ? 0 : left.Hour);
                leftMinute = left.Minute;
                leftSecond = left.Second;

                // "5 to 6pm" is 17-18, but "2:30 to 2:15 pm" crosses noon, so keep the morning reading when it must
                int leftMinutes = leftHour * 60 + (leftMinute < 0 ? 0 : leftMinute);
                int endMinutes  = endHour  * 60 + (endMinute  < 0 ? 0 : endMinute);

                if (leftMinutes > endMinutes && sharedAmPm == 1 && left.Hour <= 12)
                {
                    leftHour = left.Hour;
                }
            }
            else
            {
                ComputeTime(leftNode, out leftHour, out leftHourAlt, out leftTwo, out leftMinute, out leftSecond);
            }

            if (leftHour < 0 || endHour < 0) return false;

            // "from 10:30 to 3" closes at three in the afternoon, not at three in the morning
            if (endTwo && endHour < leftHour && endHourAlt > leftHour)
            {
                endHour = endHourAlt;
                endTwo  = false;
            }
            else if (endHour < leftHour && right.AmPm == 0 && endHour + 12 > leftHour)
            {
                // "10am-12am" runs forward to noon; midnight would put the range back to front
                endHour += 12;
            }

            range.StartHour   = leftHour;
            range.StartMinute = leftMinute < 0 ? 0 : leftMinute;
            range.StartSecond = leftSecond < 0 ? 0 : leftSecond;
            range.EndHour     = endHour;
            range.EndMinute   = endMinute < 0 ? 0 : endMinute;
            range.EndSecond   = endSecond < 0 ? 0 : endSecond;
            range.StartTimex  = TimexOfTime(leftHour, leftMinute, leftSecond);
            range.EndTimex    = TimexOfTime(endHour, endMinute, endSecond);
            range.Timex       = $"({range.StartTimex},{range.EndTimex},{ClockSpanTimex(range)})";

            if (leftTwo && endTwo)
            {
                alternate.StartHour   = leftHourAlt;
                alternate.StartMinute = range.StartMinute;
                alternate.StartSecond = range.StartSecond;
                alternate.EndHour     = endHourAlt;
                alternate.EndMinute   = range.EndMinute;
                alternate.EndSecond   = range.EndSecond;
                alternate.StartTimex  = TimexOfTime(leftHourAlt, leftMinute, leftSecond);
                alternate.EndTimex    = TimexOfTime(endHourAlt, endMinute, endSecond);
                alternate.Timex       = $"({alternate.StartTimex},{alternate.EndTimex},{ClockSpanTimex(alternate)})";
                hasAlternate          = true;
            }

            return true;
        }

        private static string ClockSpanTimex(ClockRange r)
        {
            int startMinutes = r.StartHour * 60 + r.StartMinute;
            int endMinutes   = r.EndHour   * 60 + r.EndMinute;
            int diff         = endMinutes - startMinutes;

            if (diff <= 0) diff += 24 * 60;

            int hours   = diff / 60;
            int minutes = diff % 60;

            if (hours > 0 && minutes > 0) return $"PT{hours}H{minutes}M";
            if (hours > 0)                return $"PT{hours}H";
            return $"PT{minutes}M";
        }

        private static string ClockValue(int hour, int minute, int second) => $"{hour:00}:{minute:00}:{second:00}";

        private void ResolveTimeRange(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            var n = At(nodeIndex);

            if (!ComputeClockRange(nodeIndex, out var range, out var alternate, out bool hasAlternate)) return;

            Emit(range);
            if (hasAlternate) Emit(alternate);

            void Emit(ClockRange r)
            {
                var value = new DateTimeResolutionValue { Timex = r.Timex, Type = "timerange", Mod = CombinedModName(n.Mod, n.InnerMod) };

                bool dropStart = false;
                bool dropEnd   = false;

                if (r.Open)
                {
                    dropStart = n.Mod == ModKind.Before || n.Mod == ModKind.Until;
                    dropEnd   = !dropStart;
                }
                else
                {
                    switch (n.Mod)
                    {
                        case ModKind.Before or ModKind.Until: dropStart = true; break;
                        case ModKind.After  or ModKind.Since:
                            r.StartHour   = r.EndHour;
                            r.StartMinute = r.EndMinute;
                            r.StartSecond = r.EndSecond;
                            dropEnd       = true;
                            break;
                    }
                }

                if (n.Mod == ModKind.Before || n.Mod == ModKind.Until)
                {
                    if (!r.Open)
                    {
                        r.EndHour   = r.StartHour;
                        r.EndMinute = r.StartMinute;
                        r.EndSecond = r.StartSecond;
                    }
                }

                if (!dropStart) value.Start = ClockValue(r.StartHour, r.StartMinute, r.StartSecond);
                if (!dropEnd)   value.End   = ClockValue(r.EndHour,   r.EndMinute,   r.EndSecond);

                values.Add(value);
            }
        }

        // ------------------------------------------------------------------ datetime ranges

        /// <summary>A range whose two ends are complete moments; a side that named no day borrows the other's.</summary>
        private void ResolveMomentRange(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            var n = At(nodeIndex);

            DateTime start, end;
            string   startTimex, endTimex;

            // "from 5 to 6pm of april 22", "between 7 and 9:30 last night" — the marked end says which
            // clock the bare one means
            int marked = At(n.Right).AmPm >= 0 ? At(n.Right).AmPm : AmPmOfPart(At(n.Right).PartOfDay);
            if (marked < 0) marked = AmPmOfPart(n.PartOfDay);

            ref var left = ref At(n.Left);

            int rightHour = At(n.Right).Hour;
            if (marked == 1 && rightHour >= 0 && rightHour < 12) rightHour += 12;

            if (marked == 1 && left.AmPm < 0 && left.PartOfDay == PartOfDayKind.None && left.Hour > 0 && left.Hour < 12
                && (rightHour < 0 || left.Hour + 12 <= rightHour))
            {
                left.AmPm = 1;
            }

            // Whichever end names a day sets the day for the other: "from 5 to 6pm of april 22"
            if (!At(n.Left).HasDate && At(n.Right).HasDate)
            {
                if (!Moment(n.Right, _reference, out end, out endTimex)) return;
                if (!Moment(n.Left, end, out start, out startTimex)) return;
            }
            else
            {
                if (!Moment(n.Left, _reference, out start, out startTimex)) return;
                if (!Moment(n.Right, start, out end, out endTimex)) return;
            }

            if (end < start) end = end.AddDays(1);

            var span  = end - start;
            int hours = (int)span.TotalHours;
            int mins  = (int)(span.TotalMinutes - hours * 60);

            string duration = mins > 0 ? $"PT{hours}H{mins}M" : $"PT{hours}H";
            string timex    = $"({startTimex},{endTimex},{duration})";

            values.Add(new DateTimeResolutionValue
            {
                Timex = timex,
                Type  = "datetimerange",
                Start = FormatDateTime(start),
                End   = FormatDateTime(end),
                Mod   = CombinedModName(n.Mod, n.InnerMod),
            });

            // "between 10 and 11:30 on 1/1/2015" — neither end said which half of the day it meant
            if (marked < 0 && left.AmPm < 0 && At(n.Right).AmPm < 0
                && left.PartOfDay == PartOfDayKind.None && At(n.Right).PartOfDay == PartOfDayKind.None
                && start.Hour > 0 && start.Hour < 12 && end.Hour > 0 && end.Hour < 12
                && !timex.Contains("XXXX", StringComparison.Ordinal))
            {
                var otherStart = start.AddHours(12);
                var otherEnd   = end.AddHours(12);

                values.Add(new DateTimeResolutionValue
                {
                    Timex = $"({startTimex.Replace($"T{start.Hour:00}", $"T{otherStart.Hour:00}", StringComparison.Ordinal)},{endTimex.Replace($"T{end.Hour:00}", $"T{otherEnd.Hour:00}", StringComparison.Ordinal)},{duration})",
                    Type  = "datetimerange",
                    Start = FormatDateTime(otherStart),
                    End   = FormatDateTime(otherEnd),
                    Mod   = CombinedModName(n.Mod, n.InnerMod),
                });
            }

            // A day neither end pinned to a year has the second reading a year on
            if (timex.Contains("XXXX", StringComparison.Ordinal))
            {
                values.Add(new DateTimeResolutionValue
                {
                    Timex = timex,
                    Type  = "datetimerange",
                    Start = FormatDateTime(start.AddYears(1)),
                    End   = FormatDateTime(end.AddYears(1)),
                    Mod   = CombinedModName(n.Mod, n.InnerMod),
                });
            }
        }

        /// <summary>Which half of the day a part of the day falls in, or -1 where it says nothing.</summary>
        private static int AmPmOfPart(PartOfDayKind part) => part switch
        {
            PartOfDayKind.Morning or PartOfDayKind.EarlyMorning or PartOfDayKind.Breakfast => 0,
            PartOfDayKind.Afternoon or PartOfDayKind.Evening or PartOfDayKind.Night
                or PartOfDayKind.Tonight or PartOfDayKind.Dinner                           => 1,
            _                                                                              => -1,
        };

        private bool Moment(int nodeIndex, DateTime fallbackDay, out DateTime moment, out string timex)
        {
            moment = default;
            timex  = null;

            if (nodeIndex < 0) return false;

            ref var n = ref At(nodeIndex);

            DateTime day = fallbackDay.Date;
            string dayTimex = FormatDate(day);

            if (n.HasDate && ComputeDate(nodeIndex, out var computedTimex, out var computedDay, out _, out _))
            {
                day      = computedDay;
                dayTimex = computedTimex;
            }

            if (!n.HasAnyTime)
            {
                moment = day;
                timex  = dayTimex;
                return true;
            }

            ComputeTime(nodeIndex, out int hour, out _, out _, out int minute, out int second);
            if (hour < 0) return false;

            int m = minute < 0 ? 0 : minute;
            int sec = second < 0 ? 0 : second;

            moment = new DateTime(day.Year, day.Month, day.Day, hour, m, sec);
            timex  = dayTimex + TimexOfTime(hour, minute, second);
            return true;
        }

        private void ResolveDateTimeRange(int nodeIndex, List<DateTimeResolutionValue> values)
        {
            var n = At(nodeIndex);

            // "next hour", "within 2h", "last minute"
            if (n.PeriodUnit != TimeUnit.None)
            {
                int count = Math.Max(1, n.PeriodCount);
                double seconds = n.PeriodUnit switch
                {
                    TimeUnit.Hour   => count * 3600d,
                    TimeUnit.Minute => count * 60d,
                    _               => count * 1d,
                };

                bool backwards = n.Relative == RelativeKind.Last || n.Relative == RelativeKind.Previous || n.Relative == RelativeKind.BeforeLast;

                var start = backwards ? _reference.AddSeconds(-seconds) : _reference;
                var end   = backwards ? _reference : _reference.AddSeconds(seconds);

                values.Add(new DateTimeResolutionValue
                {
                    Timex = $"({FormatDate(start)}{TimexOfTime(start.Hour, start.Minute, start.Second)},{FormatDate(end)}{TimexOfTime(end.Hour, end.Minute, end.Second)},{DurationTimexOf(n.PeriodUnit, count)})",
                    Type  = "datetimerange",
                    Start = FormatDateTime(start),
                    End   = FormatDateTime(end),
                    Mod   = CombinedModName(n.Mod, n.InnerMod),
                });
                return;
            }

            if (n.ChildrenAreMoments)
            {
                ResolveMomentRange(nodeIndex, values);
                return;
            }

            DateTime day;
            DateTime secondDay;
            bool     twoDays;
            string   dayTimex;

            if (ComputeDate(nodeIndex, out var dateTimex, out var d1, out var d2, out twoDays))
            {
                day       = d1;
                secondDay = d2;
                dayTimex  = dateTimex;
            }
            else
            {
                day       = _reference.Date;
                secondDay = day;
                twoDays   = false;
                dayTimex  = FormatDate(day);
            }

            if (!ComputeClockRange(nodeIndex, out var range, out var alternate, out bool hasAlternate)) return;

            Emit(range, day);
            if (twoDays) Emit(range, secondDay);

            if (hasAlternate)
            {
                Emit(alternate, day);
                if (twoDays) Emit(alternate, secondDay);
            }

            void Emit(ClockRange r, DateTime day)
            {
                string timex;

                if (r.StartTimex is null)
                {
                    timex = dayTimex;   // "mid today" is named by the day it cuts
                }
                else if ((n.PartOfDay != PartOfDayKind.None && n.Left < 0) || r.Open)
                {
                    timex = dayTimex + r.Timex;
                }
                else
                {
                    timex = $"({dayTimex}{r.StartTimex},{dayTimex}{r.EndTimex},{ClockSpanTimex(r)})";
                }

                var start = day.AddHours(r.StartHour).AddMinutes(r.StartMinute).AddSeconds(r.StartSecond);
                var end   = day.AddHours(r.EndHour).AddMinutes(r.EndMinute).AddSeconds(r.EndSecond);

                var value = new DateTimeResolutionValue { Timex = timex, Type = "datetimerange", Mod = CombinedModName(n.Mod, n.InnerMod) };

                bool dropStart = n.Mod == ModKind.Before || n.Mod == ModKind.Until;
                bool dropEnd   = n.Mod == ModKind.After  || n.Mod == ModKind.Since;

                if (r.Open)
                {
                    if (dropStart) { end = start; }
                    else           { dropEnd = true; }
                }

                if (!dropStart) value.Start = FormatDateTime(start);
                if (!dropEnd)   value.End   = FormatDateTime(end);

                values.Add(value);
            }
        }
    }
}
