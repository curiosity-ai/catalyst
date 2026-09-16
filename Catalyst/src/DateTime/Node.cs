using System;
using System.Globalization;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The per-unit totals of a duration, kept separate so the TIMEX keeps the shape the text used.</summary>
    public struct DurationParts
    {
        public const double SecondsPerYear    = 31536000d;
        public const double SecondsPerMonth   = 2592000d;
        public const double SecondsPerWeek    = 604800d;
        public const double SecondsPerDay     = 86400d;
        public const double SecondsPerWeekend = 172800d;
        public const double SecondsPerHour    = 3600d;
        public const double SecondsPerMinute  = 60d;

        public double Years;
        public double Months;
        public double Weeks;
        public double Days;
        public double Weekends;
        public double Hours;
        public double Minutes;
        public double Seconds;
        public bool   Any;

        public readonly bool IsDateOnly => Hours == 0 && Minutes == 0 && Seconds == 0;

        public void Add(TimeUnit unit, double amount)
        {
            switch (unit)
            {
                case TimeUnit.Year:        Years    += amount;      break;
                case TimeUnit.Decade:      Years    += amount * 10; break;
                case TimeUnit.Century:     Years    += amount * 100;break;
                case TimeUnit.Month:       Months   += amount;      break;
                case TimeUnit.Quarter:     Months   += amount * 3;  break;
                case TimeUnit.Week:        Weeks    += amount;      break;
                case TimeUnit.WorkWeek:    Weeks    += amount;      break;
                case TimeUnit.Fortnight:   Weeks    += amount * 2;  break;
                case TimeUnit.Weekend:     Weekends += amount;      break;
                case TimeUnit.Day:         Days     += amount;      break;
                case TimeUnit.Night:       Days     += amount;      break;
                case TimeUnit.BusinessDay: Days     += amount;      break;
                case TimeUnit.Hour:        Hours    += amount;      break;
                case TimeUnit.Minute:      Minutes  += amount;      break;
                case TimeUnit.Second:      Seconds  += amount;      break;
                default: return;
            }

            Any = true;
        }

        public readonly double TotalSeconds =>
              Years    * SecondsPerYear
            + Months   * SecondsPerMonth
            + Weeks    * SecondsPerWeek
            + Weekends * SecondsPerWeekend
            + Days     * SecondsPerDay
            + Hours    * SecondsPerHour
            + Minutes  * SecondsPerMinute
            + Seconds;

        public static string Fmt(double v) => v.ToString("0.############", System.Globalization.CultureInfo.InvariantCulture);

        public readonly string ToTimex()
        {
            var sb = new System.Text.StringBuilder(16);
            sb.Append('P');

            if (Years    != 0) { sb.Append(Fmt(Years)).Append('Y'); }
            if (Months   != 0) { sb.Append(Fmt(Months)).Append('M'); }
            if (Weeks    != 0) { sb.Append(Fmt(Weeks)).Append('W'); }
            if (Weekends != 0) { sb.Append(Fmt(Weekends)).Append("WE"); }
            if (Days     != 0) { sb.Append(Fmt(Days)).Append('D'); }

            if (!IsDateOnly)
            {
                sb.Append('T');
                if (Hours   != 0) { sb.Append(Fmt(Hours)).Append('H'); }
                if (Minutes != 0) { sb.Append(Fmt(Minutes)).Append('M'); }
                if (Seconds != 0) { sb.Append(Fmt(Seconds)).Append('S'); }
            }

            return sb.ToString();
        }
    }

    public enum NodeKind : byte
    {
        None = 0,
        Date,
        Time,
        DateTime,
        DateRange,
        TimeRange,
        DateTimeRange,
        Duration,
        Set,
    }

    /// <summary>
    /// A parsed construct. Nodes live in an arena owned by the parser and reference each other by index,
    /// so a range can hold its two endpoints without any of it being heap-allocated per match.
    /// </summary>
    public struct Node
    {
        public const int Unspecified = -1;

        public NodeKind Kind;
        public int      LexStart;
        public int      LexEnd;      // exclusive
        public int      CharStart;
        public int      CharEnd;     // exclusive

        // Date
        public int          Year;
        public int          Month;
        public int          Day;
        public int          Weekday;
        public RelativeKind Relative;
        public int          OffsetDays;
        public int          OffsetWeeks;
        public int          OffsetMonths;
        public int          OffsetYears;
        public HolidayKind  Holiday;
        public SeasonKind   Season;
        /// <summary>The node this one is measured from ("3 days after january 12th"), or -1.</summary>
        public int          Anchor;
        /// <summary>The duration a clock range runs for ("for 2 hours from 2pm"), or -1.</summary>
        public int          RangeDuration;

        // Period
        public TimeUnit PeriodUnit;
        public int      PeriodCount;
        public int      OrdinalInPeriod;   // "first week of X" -> 1
        public bool     OrdinalFromEnd;    // "the last week of X"
        public int      WeekOfYear;
        public int      Quarter;
        public int      HalfOfYear;
        public int      Decade;
        public int      Century;
        public bool     BusinessDays;
        public int      FiscalKind;        // -1 none, 0 calendar, 1 fiscal, 2 school

        // Time
        public int           Hour;
        public int           Minute;
        public int           Second;
        public int           AmPm;          // -1 unknown, 0 am, 1 pm
        public PartOfDayKind PartOfDay;

        // Duration
        public DurationParts Duration;
        public double   DurationSeconds;
        public string   DurationTimex;

        // Range
        public int  Left;
        public int  Right;
        /// <summary>True when Left and Right are whole moments rather than two readings of the same clock.</summary>
        public bool ChildrenAreMoments;

        /// <summary>True where the day was written with an article ("monday the 26th"), which names one day.</summary>
        public bool DefiniteDay;

        public ModKind Mod;
        /// <summary>A narrowing modifier the bounding one wraps: the "mid" of "after mid may".</summary>
        public ModKind InnerMod;

        // Set
        public int      SetInterval;
        public TimeUnit SetUnit;

        public static Node Create(NodeKind kind)
        {
            return new Node
            {
                Kind            = kind,
                Year            = Unspecified,
                Month           = Unspecified,
                Day             = Unspecified,
                Weekday         = Unspecified,
                Hour            = Unspecified,
                Minute          = Unspecified,
                Second          = Unspecified,
                AmPm            = Unspecified,
                Anchor          = Unspecified,
                RangeDuration   = Unspecified,
                Left            = Unspecified,
                Right           = Unspecified,
                OrdinalInPeriod = Unspecified,
                WeekOfYear      = Unspecified,
                Quarter         = Unspecified,
                HalfOfYear      = Unspecified,
                Decade          = Unspecified,
                Century         = Unspecified,
                FiscalKind      = Unspecified,
                SetInterval     = 1,
            };
        }

        public readonly bool HasDate    => Year >= 0 || Month >= 0 || Day >= 0 || Weekday >= 0 || Holiday != HolidayKind.None || OffsetDays != 0 || OffsetWeeks != 0 || OffsetMonths != 0 || OffsetYears != 0 || Relative != RelativeKind.None;
        public readonly bool HasAnyTime => Hour >= 0;
    }
}
