using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The English vocabulary. Every word the grammar can act on is listed here exactly once.</summary>
    public static class EnglishLexicon
    {
        private static readonly Lazy<Lexicon> _usOrder    = new Lazy<Lexicon>(() => Build(dayMonthOrder: false), true);
        private static readonly Lazy<Lexicon> _dayMonth   = new Lazy<Lexicon>(() => Build(dayMonthOrder: true),  true);

        /// <param name="dayMonthOrder">true for the rest of the world (22/04 is 22 April), false for US English (04/22).</param>
        public static Lexicon Get(bool dayMonthOrder) => dayMonthOrder ? _dayMonth.Value : _usOrder.Value;

        private static Lexicon Build(bool dayMonthOrder)
        {
            var b = new LexiconBuilder();

            AddMonths(b);
            AddWeekdays(b);
            AddNumbers(b);
            AddUnits(b);
            AddRelatives(b);
            AddPartsOfDay(b);
            AddConnectors(b);
            AddModifiers(b);
            AddSets(b);
            AddSeasons(b);
            AddHolidays(b);

            return b.Build(Language.English, dayMonthOrder, partNamedWithOf: true);
        }

        private static void AddMonths(LexiconBuilder b)
        {
            b.Add(TermKind.Month,  1, "january", "jan", "jan.");
            b.Add(TermKind.Month,  2, "february", "feb", "febr");
            b.Add(TermKind.Month,  3, "march", "mar");
            b.Add(TermKind.Month,  4, "april", "apr");
            b.Add(TermKind.Month,  5, "may");
            b.Add(TermKind.Month,  6, "june", "jun");
            b.Add(TermKind.Month,  7, "july", "jul");
            b.Add(TermKind.Month,  8, "august", "aug");
            b.Add(TermKind.Month,  9, "september", "sep", "sept", "spt");
            b.Add(TermKind.Month, 10, "october", "oct");
            b.Add(TermKind.Month, 11, "november", "nov");
            b.Add(TermKind.Month, 12, "december", "dec");
        }

        private static void AddWeekdays(LexiconBuilder b)
        {
            b.Add(TermKind.Weekday, 0, "sunday", "sundays", "sun", "suns");
            b.Add(TermKind.Weekday, 1, "monday", "mondays", "mon", "mons");
            b.Add(TermKind.Weekday, 2, "tuesday", "tuesdays", "tue", "tues", "tuesd", "tus");
            b.Add(TermKind.Weekday, 3, "wednesday", "wednesdays", "wed", "weds", "wedn", "wednes");
            b.Add(TermKind.Weekday, 4, "thursday", "thursdays", "thu", "thur", "thurs", "thus");
            b.Add(TermKind.Weekday, 5, "friday", "fridays", "fri", "fris");
            b.Add(TermKind.Weekday, 6, "saturday", "saturdays", "sat", "sats");
        }

        private static void AddNumbers(LexiconBuilder b)
        {
            b.Add(TermKind.Cardinal,  0, "zero");
            b.Add(TermKind.Cardinal,  1, "one");
            b.Add(TermKind.Cardinal,  2, "two");
            b.Add(TermKind.Cardinal,  3, "three");
            b.Add(TermKind.Cardinal,  4, "four");
            b.Add(TermKind.Cardinal,  5, "five");
            b.Add(TermKind.Cardinal,  6, "six");
            b.Add(TermKind.Cardinal,  7, "seven");
            b.Add(TermKind.Cardinal,  8, "eight");
            b.Add(TermKind.Cardinal,  9, "nine");
            b.Add(TermKind.Cardinal, 10, "ten");
            b.Add(TermKind.Cardinal, 11, "eleven");
            b.Add(TermKind.Cardinal, 12, "twelve");
            b.Add(TermKind.Cardinal, 13, "thirteen");
            b.Add(TermKind.Cardinal, 14, "fourteen");
            b.Add(TermKind.Cardinal, 15, "fifteen");
            b.Add(TermKind.Cardinal, 16, "sixteen");
            b.Add(TermKind.Cardinal, 17, "seventeen");
            b.Add(TermKind.Cardinal, 18, "eighteen");
            b.Add(TermKind.Cardinal, 19, "nineteen");
            b.Add(TermKind.Cardinal, 20, "twenty");
            b.Add(TermKind.Cardinal, 30, "thirty");
            b.Add(TermKind.Cardinal, 40, "forty", "fourty");
            b.Add(TermKind.Cardinal, 50, "fifty");
            b.Add(TermKind.Cardinal, 60, "sixty");
            b.Add(TermKind.Cardinal, 70, "seventy");
            b.Add(TermKind.Cardinal, 80, "eighty");
            b.Add(TermKind.Cardinal, 90, "ninety");

            b.Add(TermKind.Multiplier,       100, "hundred", "hundreds");
            b.Add(TermKind.Multiplier,      1000, "thousand", "thousands");
            b.Add(TermKind.Multiplier,   1000000, "million", "millions");

            b.Add(TermKind.Ordinal,  1, "first");
            b.Add(new TermInfo(TermKind.Ordinal, 2, TermKind.Unit, (int)TimeUnit.Second), "second");
            b.Add(TermKind.Ordinal,  3, "third");
            b.Add(TermKind.Ordinal,  4, "fourth");
            b.Add(TermKind.Ordinal,  5, "fifth");
            b.Add(TermKind.Ordinal,  6, "sixth");
            b.Add(TermKind.Ordinal,  7, "seventh");
            b.Add(TermKind.Ordinal,  8, "eighth");
            b.Add(TermKind.Ordinal,  9, "ninth");
            b.Add(TermKind.Ordinal, 10, "tenth");
            b.Add(TermKind.Ordinal, 11, "eleventh");
            b.Add(TermKind.Ordinal, 12, "twelfth", "twelveth");
            b.Add(TermKind.Ordinal, 13, "thirteenth");
            b.Add(TermKind.Ordinal, 14, "fourteenth");
            b.Add(TermKind.Ordinal, 15, "fifteenth");
            b.Add(TermKind.Ordinal, 16, "sixteenth");
            b.Add(TermKind.Ordinal, 17, "seventeenth");
            b.Add(TermKind.Ordinal, 18, "eighteenth");
            b.Add(TermKind.Ordinal, 19, "nineteenth");
            b.Add(TermKind.Ordinal, 20, "twentieth");
            b.Add(TermKind.Ordinal, 30, "thirtieth");
            b.Add(TermKind.Ordinal, 40, "fortieth");
            b.Add(TermKind.Ordinal, 50, "fiftieth");
            b.Add(TermKind.Ordinal, 60, "sixtieth");
            b.Add(TermKind.Ordinal, 70, "seventieth");
            b.Add(TermKind.Ordinal, 80, "eightieth");
            b.Add(TermKind.Ordinal, 90, "ninetieth");

            b.Add(TermKind.OrdinalSuffix, "st", "nd", "rd", "th");

            // Decades named as a word: "the nineties"
            b.Add(TermKind.Decade, 1920, "twenties");
            b.Add(TermKind.Decade, 1930, "thirties");
            b.Add(TermKind.Decade, 1940, "forties");
            b.Add(TermKind.Decade, 1950, "fifties");
            b.Add(TermKind.Decade, 1960, "sixties");
            b.Add(TermKind.Decade, 1970, "seventies");
            b.Add(TermKind.Decade, 1980, "eighties");
            b.Add(TermKind.Decade, 1990, "nineties");

            b.Add(TermKind.HalfWord,    "half");
            b.Add(new TermInfo(TermKind.QuarterWord, 0, TermKind.Unit, (int)TimeUnit.Quarter), "quarter", "quarters");
            b.Add(TermKind.Several, 3, "few", "several", "some", "many");
            b.Add(TermKind.Several, 2, "couple");
            b.Add(TermKind.Several, 1, "another");
        }

        private static void AddUnits(LexiconBuilder b)
        {
            b.Add(TermKind.Unit, (int)TimeUnit.Second,      "seconds", "sec", "secs", "s.", "segs");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,      "minute", "minutes", "min", "mins");
            b.Add(TermKind.Unit, (int)TimeUnit.Hour,        "hour", "hours", "hr", "hrs");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.QuarterMarker, 2), "h");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,         "day", "days", "d");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,        "week", "weeks", "wk", "wks", "w");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight,   "fortnight", "fortnights");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,       "month", "months");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,        "year", "years", "yr", "yrs", "y");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,      "decade", "decades");
            b.Add(TermKind.Unit, (int)TimeUnit.Century,     "century", "centuries");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend,     "weekend", "weekends");
            b.Add(TermKind.Unit, (int)TimeUnit.WorkWeek,    "workweek", "workweeks", "working week", "working weeks", "work week", "work weeks", "business week");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "night", "nights");

            b.Add(TermKind.BusinessDay, "business", "working", "work", "weekday", "weekdays");

            b.Add(TermKind.Fiscal, 0, "calendar", "cy");
            b.Add(TermKind.Fiscal, 1, "fiscal", "fy");
            b.Add(TermKind.Fiscal, 2, "school", "sy", "academic");

            b.Add(TermKind.QuarterMarker, 4, "q");
            b.Add(TermKind.WeekMarker,       "wk#", "w/c");
        }

        private static void AddRelatives(LexiconBuilder b)
        {
            b.Add(TermKind.Relative, (int)RelativeKind.This,      "this", "that");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,      "next");
            b.Add(TermKind.Relative, (int)RelativeKind.Coming,    "coming", "upcoming");
            b.Add(TermKind.Relative, (int)RelativeKind.Following, "following");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous,  "previous", "prior", "preceding");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,   "current", "same", "present");
            b.Add(TermKind.Relative, (int)RelativeKind.Last, "last");
            b.Add(new TermInfo(TermKind.Relative, (int)RelativeKind.JustPast, TermKind.PastWord), "past");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,     "today", "otd");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow,  "tomorrow", "tomorow", "tmr", "tmrw", "tomm", "tommorow", "tommorrow");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday, "yesterday", "yday", "ytd.");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,       "now", "currently", "instantly", "immediately");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "overmorrow", "the day after tomorrow", "day after tomorrow");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.TheDay,              "the day");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.NextDay,             "the next day", "next day", "the day after", "the following day");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.PriorDay,            "the day before", "the previous day", "prior day");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "the day before yesterday", "day before yesterday");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "right now", "at the moment", "at the minute", "at present", "at this time", "at the present time");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.EndOfDay,           "end of day", "end of the day", "eod", "the eod");

            b.Add(new TermInfo(TermKind.Ago, 0, TermKind.Mod, (int)ModKind.Earlier), "earlier");
            b.Add(TermKind.Ago,     "ago", "before now");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later), "later");
            b.Add(TermKind.FromNow, "hence", "afterwards", "after now", "from now", "in the future");
        }

        private static void AddPartsOfDay(LexiconBuilder b)
        {
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "morning", "mornings");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "afternoon", "afternoons");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "evening", "evenings");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "noon", "noonish", "midday");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "midnight", "mid night", "mid-night");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.DayTime,   "daytime", "day time", "day-time");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "tonight", "tonite", "overnight");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "lunch", "lunchtime");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "dinner", "dinnertime", "suppertime", "supper");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "breakfast");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Brunch,    "brunch");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Business,  "business hours", "working hours", "office hours");

            b.Add(TermKind.AmPm, 0, "am", "a.m", "a.m.", "a");
            b.Add(TermKind.AmPm, 1, "pm", "p.m", "p.m.", "p");

            b.Add(TermKind.OClock, "oclock", "o'clock", "o'", "clock", "hours sharp");
        }

        private static void AddConnectors(LexiconBuilder b)
        {
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "to", "til");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.Mod, (int)ModKind.Before), "till", "until", "untill", "thru", "through");
            b.Add(TermKind.Connector, "and", "&");

            b.Add(TermKind.RangeStart, 0, "from", "starting", "beginning", "commencing", "starting from", "beginning on", "starting on", "beginning from");
            b.Add(TermKind.RangeStart, 1, "between");

            b.Add(TermKind.Filler, "of", "on", "in", "at", "the", "a", "an", "for", "during", "within", "into", "or", "s");
        }

        private static void AddModifiers(LexiconBuilder b)
        {
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "before", "by", "no later than", "not later than", "earlier than", "prior to");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.After,  TermKind.PastWord), "after");
            b.Add(TermKind.Mod, (int)ModKind.After,     "later than", "greater than", "starting after", "no earlier than");
            b.Add(TermKind.Mod, (int)ModKind.Less,      "less than", "fewer than");
            b.Add(TermKind.Mod, (int)ModKind.More,      "more than");
            b.Add(TermKind.Mod, (int)ModKind.Since,     "since", "since then", "as of");
            b.Add(TermKind.Mod, (int)ModKind.Start,     "start of", "beginning of");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Start, TermKind.RangeStart, 0), "start", "beginning");
            b.Add(TermKind.Mod, (int)ModKind.End,       "end of", "end");
            b.Add(TermKind.Mod, (int)ModKind.Mid,       "mid", "middle of", "mid of");
            b.Add(TermKind.Mod, (int)ModKind.Early,     "early");
            b.Add(TermKind.Mod, (int)ModKind.Late,      "late");
            b.Add(TermKind.Mod, (int)ModKind.Since,     "as early as");
            b.Add(TermKind.Mod, (int)ModKind.Until,     "as late as");
            b.Add(TermKind.Mod, (int)ModKind.OrLater,   "or later", "and later", "and after", "and greater", "or greater", "or after");
            b.Add(TermKind.Mod, (int)ModKind.OrEarlier, "or earlier", "and earlier", "or before", "and before");

            // "about" is not read as an approximation by the suite; "around" and "circa" are
            b.Add(TermKind.Approx, "around", "circa", "approximately", "roughly", "nearly", "almost", "ish", "sometime around");
        }

        private static void AddSets(LexiconBuilder b)
        {
            b.Add(TermKind.SetPrefix, 0, "every", "each");
            b.Add(TermKind.SetPrefix, 1, "any", "all");

            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "daily", "everyday", "every day", "nightly");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "weekly", "hebdomadal");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "monthly");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "yearly", "annually", "annual");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "hourly");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "quarterly");
            b.Add(new TermInfo(TermKind.SetFrequency, (int)TimeUnit.Month, TermKind.Multiplier, 2), "bi monthly", "bimonthly");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "semi monthly");
            b.Add(new TermInfo(TermKind.SetFrequency, (int)TimeUnit.Week,  TermKind.Multiplier, 2), "bi weekly", "biweekly");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.HalfYear, "semi annually", "semiannually", "semiannual", "semi annual", "biannual", "biannually");
        }

        private static void AddSeasons(LexiconBuilder b)
        {
            b.Add(TermKind.Season, (int)SeasonKind.Spring, "spring", "springtime");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "summer", "summertime");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "fall", "autumn");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "winter", "wintertime");
        }

        private static void AddHolidays(LexiconBuilder b)
        {
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,        "new year", "new years", "new year's day", "new years day", "new year day", "newyear", "yuandan");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,     "new year's eve", "new years eve", "new year eve", "newyearseve");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,      "christmas", "christmas day", "xmas", "christmasday");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve,   "christmas eve", "xmas eve", "christmaseve");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,         "easter", "easter day", "easter sunday");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday,   "easter monday");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,     "good friday");
            b.Add(TermKind.Holiday, (int)HolidayKind.Thanksgiving,   "thanksgiving", "thanksgiving day", "thanksgivingday");
            b.Add(TermKind.Holiday, (int)HolidayKind.BlackFriday,    "black friday", "blackfriday");
            b.Add(TermKind.Holiday, (int)HolidayKind.CyberMonday,    "cyber monday", "cybermonday");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,      "halloween", "halloween", "all hallows day", "all hallow day");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,     "valentines day", "valentine's day", "valentines", "valentinesday");
            b.Add(TermKind.Holiday, (int)HolidayKind.AprilFools,     "april fools day", "april fool's day", "april fools", "aprilfools");
            b.Add(TermKind.Holiday, (int)HolidayKind.IndependenceDay,"independence day", "independenceday", "fourth of july", "4th of july");
            b.Add(TermKind.Holiday, (int)HolidayKind.MemorialDay,    "memorial day", "memorialday");
            b.Add(TermKind.Holiday, (int)HolidayKind.LaborDay,       "labor day", "labour day", "labourday", "laborday");
            b.Add(TermKind.Holiday, (int)HolidayKind.ColumbusDay,    "columbus day", "columbusday");
            b.Add(TermKind.Holiday, (int)HolidayKind.VeteransDay,    "veterans day", "veteransday");
            b.Add(TermKind.Holiday, (int)HolidayKind.MartinLutherKingDay, "martin luther king day", "mlk day", "martin luther king jr day");
            b.Add(TermKind.Holiday, (int)HolidayKind.PresidentsDay,  "presidents day", "president's day", "presidentsday");
            b.Add(TermKind.Holiday, (int)HolidayKind.StPatricksDay,  "saint patrick", "saint patrick's day", "st patrick's day", "st patricks day", "saint patricks day");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,     "mothers day", "mother's day", "mothersday");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,     "fathers day", "father's day", "fathersday");
            b.Add(TermKind.Holiday, (int)HolidayKind.EarthDay,       "earth day", "earthday");
            b.Add(TermKind.Holiday, (int)HolidayKind.Juneteenth,     "juneteenth", "juneteenth day");
            b.Add(TermKind.Holiday, (int)HolidayKind.FreedomDay,     "freedom day", "freedomday");
            b.Add(TermKind.Holiday, (int)HolidayKind.JubileeDay,     "jubilee day", "jubileeday");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "international workers' day", "international workers day", "may day", "workers day");
            b.Add(TermKind.Holiday, (int)HolidayKind.Groundhog,      "groundhog day", "groundhogday");
            b.Add(TermKind.Holiday, (int)HolidayKind.Boxing,         "boxing day", "boxingday");
            b.Add(TermKind.Holiday, (int)HolidayKind.EidAlFitr,      "eid al fitr", "eid al-fitr", "eid ul fitr");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,      "all saints day", "all saints' day", "all souls day");
            b.Add(TermKind.Holiday, (int)HolidayKind.GermanUnityDay, "german unity day");
            b.Add(TermKind.Holiday, (int)HolidayKind.BastilleDay,    "bastille day");
            b.Add(TermKind.Holiday, (int)HolidayKind.CanadaDay,      "canada day");
            b.Add(TermKind.Holiday, (int)HolidayKind.AustraliaDay,   "australia day");
            b.Add(TermKind.Holiday, (int)HolidayKind.AnzacDay,       "anzac day");
        }
    }
}
