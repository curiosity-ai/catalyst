using System;
using System.Collections.Generic;

namespace Catalyst.DateTimeRecognition
{
    public enum DateTimeEntityKind : byte
    {
        Date = 0,
        Time,
        DateTime,
        DateRange,
        TimeRange,
        DateTimeRange,
        Duration,
        Set,
    }

    /// <summary>One reading of a recognised expression, in the same shape Microsoft.Recognizers.Text produces.</summary>
    public sealed class DateTimeResolutionValue
    {
        public string Timex { get; set; }
        public string Type  { get; set; }
        public string Value { get; set; }
        public string Start { get; set; }
        public string End   { get; set; }
        public string Mod   { get; set; }

        public Dictionary<string, string> ToDictionary()
        {
            var d = new Dictionary<string, string>(6);

            if (Timex is object) d["timex"] = Timex;
            if (Type  is object) d["type"]  = Type;
            if (Value is object) d["value"] = Value;
            if (Start is object) d["start"] = Start;
            if (End   is object) d["end"]   = End;
            if (Mod   is object) d["Mod"]   = Mod;

            return d;
        }
    }

    /// <summary>A recognised date/time expression and where it was found.</summary>
    public sealed class DateTimeEntity
    {
        public string                          Text   { get; set; }
        /// <summary>Index of the first character, inclusive.</summary>
        public int                             Start  { get; set; }
        /// <summary>Index of the last character, inclusive — the convention Microsoft.Recognizers.Text uses.</summary>
        public int                             End    { get; set; }
        public DateTimeEntityKind              Kind   { get; set; }
        public List<DateTimeResolutionValue>   Values { get; set; }

        public string TypeName => Kind switch
        {
            DateTimeEntityKind.Date          => "datetimeV2.date",
            DateTimeEntityKind.Time          => "datetimeV2.time",
            DateTimeEntityKind.DateTime      => "datetimeV2.datetime",
            DateTimeEntityKind.DateRange     => "datetimeV2.daterange",
            DateTimeEntityKind.TimeRange     => "datetimeV2.timerange",
            DateTimeEntityKind.DateTimeRange => "datetimeV2.datetimerange",
            DateTimeEntityKind.Duration      => "datetimeV2.duration",
            DateTimeEntityKind.Set           => "datetimeV2.set",
            _                                => "datetimeV2",
        };

        public override string ToString() => $"{TypeName} [{Start}..{End}] \"{Text}\"";
    }
}
