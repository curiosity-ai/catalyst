using System;
using Mosaik.Core;
using System.Collections.Frozen;
using System.Collections.Generic;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>
    /// The semantic class a lexicon word belongs to. The grammar switches on this instead of matching
    /// character patterns, which is what lets the whole engine run without a single regular expression.
    /// </summary>
    public enum TermKind : byte
    {
        None = 0,
        Month,              // Value = 1..12
        Weekday,            // Value = 0..6, Sunday = 0
        Cardinal,           // Value = the number itself
        Ordinal,            // Value = the number itself
        Multiplier,         // hundred / thousand / million, Value = the multiplier
        Unit,               // Value = (int)TimeUnit
        Relative,           // Value = (int)RelativeKind
        SpecialDay,         // Value = (int)SpecialDayKind
        PartOfDay,          // Value = (int)PartOfDayKind
        AmPm,               // Value = 0 (am) or 1 (pm)
        Connector,          // to / till / until / through / and
        AndWord,            // the language's "and" - joins a between-range, but cannot open one alone
        LengthWord,         // "dure", "durante" - says the number that follows counts a length, not a clock
        RangeStart,         // from / between
        Mod,                // Value = (int)ModKind
        SetFrequency,       // Value = (int)TimeUnit, from daily / weekly / ...
        SetPrefix,          // every / each
        Filler,             // of / on / in / at / the / a / an — skippable glue
        Holiday,            // Value = (int)HolidayKind
        Season,             // Value = (int)SeasonKind
        OClock,             // o'clock / oclock
        HalfWord,           // half
        QuarterWord,        // quarter
        PastWord,           // past / after, in "ten past nine"
        ToWord,             // to / before / til, in "quarter to nine"
        Approx,             // around / circa / about / approximately / ish
        Ago,                // ago
        FromNow,            // later / from now / hence / afterwards
        Several,            // a few / several / some / couple
        Whole,              // whole / entire — "the whole day" is one day
        Article,            // the / de / el — glue that can head a phrase, unlike a preposition
        BusinessDay,        // business / working / work (day)
        OrdinalSuffix,      // st / nd / rd / th, following a digit
        Fiscal,             // fiscal / calendar / school, qualifying "year"
        QuarterMarker,      // the "q" of q1, or "h" of h2 (Value = periods per year)
        WeekMarker,         // the "week" of "week 27"
        Decade,             // "the nineties", Value = the first year of the decade
        ClockPrefix,        // the word that introduces a clock: "at 5", "a las 5", "um 8"
        InPrefix,           // "in 3 days" / "en 2 semanas"; Value 1 means the "within" sense
    }

    public enum TimeUnit : byte
    {
        None = 0,
        Second,
        Minute,
        Hour,
        Day,
        Week,
        Fortnight,
        Month,
        Quarter,
        Year,
        Decade,
        Century,
        Weekend,
        WorkWeek,
        HalfYear,
        BusinessDay,
        Night,
    }

    public enum RelativeKind : byte
    {
        None = 0,
        This,
        Next,
        Last,
        Coming,     // coming / upcoming — same resolution as Next but a distinct surface form
        Following,  // following — Next
        Previous,   // previous — Last
        Current,    // current / same — This
        JustPast,   // past — the most recent occurrence, which may be in this week
        AfterNext,  // "übernächste", "overmorgen week" — two units on, not one
        BeforeLast, // "vorletzte", "eergisteren week" — two units back, not one
    }

    public enum SpecialDayKind : byte
    {
        None = 0,
        Today,
        Tomorrow,
        Yesterday,
        Now,
        DayAfterTomorrow,
        DayBeforeYesterday,
        TheDay,             // "the day" == today
        NextDay,            // "the next day" / "the day after"
        PriorDay,           // "the day before"
        EndOfDay,
    }

    public enum PartOfDayKind : byte
    {
        None = 0,
        Morning,
        Afternoon,
        Evening,
        Night,
        Noon,
        Midnight,
        DayTime,
        MidDay,
        Business,       // business hours
        EarlyMorning,
        Dawn,
        LateNight,
        Tonight,
        Lunch,
        Dinner,
        Breakfast,
        Brunch,
    }

    public enum SeasonKind : byte
    {
        None = 0,
        Spring,
        Summer,
        Fall,
        Winter,
    }

    public enum ModKind : byte
    {
        None = 0,
        Before,         // "before 2010"       -> mod "before"
        After,          // "after 2010"        -> mod "after"
        Since,          // "since 2010"        -> mod "since"
        Until,          // "until april 27th"  -> mod "before"
        Start,          // "beginning of"      -> mod "start"
        End,            // "end of"            -> mod "end"
        Mid,            // "mid may"
        Approx,         // "around"
        Less,           // "less than"
        More,           // "more than"
        Early,          // "early september"
        Late,           // "late july"
        OrLater,        // "2018 or later"
        OrEarlier,      // "2018 or earlier"
        RefUndef,       // "the same week" — the period the reference moment falls in
        Earlier,        // "earlier this month" — the first part of the period
        Later,          // "later this month" — the last part of the period
    }

    public enum HolidayKind : byte
    {
        None = 0,
        NewYear,
        NewYearEve,
        Christmas,
        ChristmasEve,
        Easter,
        EasterMonday,
        GoodFriday,
        Thanksgiving,
        BlackFriday,
        Halloween,
        Valentines,
        AprilFools,
        IndependenceDay,
        MemorialDay,
        LaborDay,
        ColumbusDay,
        VeteransDay,
        MartinLutherKingDay,
        PresidentsDay,
        StPatricksDay,
        MothersDay,
        FathersDay,
        EarthDay,
        Juneteenth,
        FreedomDay,
        JubileeDay,
        InternationalWorkersDay,
        Groundhog,
        Boxing,
        CyberMonday,
        WhiteLoverDay,
        Yuandan,
        EidAlFitr,
        Ramadan,
        Diwali,
        Hanukkah,
        Passover,
        RoshHashanah,
        YomKippur,
        AllSaints,
        GermanUnityDay,
        BastilleDay,
        CanadaDay,
        AustraliaDay,
        AnzacDay,
        Epiphany,
        PalmSunday,
        HolySaturday,
        Pentecost,
        CarnivalSaturday,
        CarnivalSunday,
        HarvestThanksgiving,
        EternitySunday,
        StBarbara,
        StJohn,
        PeterAndPaul,
        ChildrensDay,
        AugsburgPeace,
        KingsDay,
        SpringStart,
        SummerStart,
        AutumnStart,
        WinterStart,
    }

    /// <summary>What a lexicon entry means: its <see cref="TermKind"/> plus a kind-specific payload.</summary>
    public readonly struct TermInfo
    {
        public readonly TermKind Kind;
        public readonly int      Value;
        /// <summary>A second reading of the same word, for the genuinely ambiguous ones ("second", "quarter", "past").</summary>
        public readonly TermKind AltKind;
        public readonly int      AltValue;

        public TermInfo(TermKind kind, int value = 0)
        {
            Kind     = kind;
            Value    = value;
            AltKind  = TermKind.None;
            AltValue = 0;
        }

        public TermInfo(TermKind kind, int value, TermKind altKind, int altValue = 0)
        {
            Kind     = kind;
            Value    = value;
            AltKind  = altKind;
            AltValue = altValue;
        }

        public bool Is(TermKind kind) => Kind == kind || AltKind == kind;

        public bool Is(TermKind kind, out int value)
        {
            if (Kind == kind)    { value = Value;    return true; }
            if (AltKind == kind) { value = AltValue; return true; }
            value = 0;
            return false;
        }

        public static readonly TermInfo Unknown = new TermInfo(TermKind.None);
    }

    /// <summary>
    /// A multi-word phrase (a holiday name, "o'clock", "the day after tomorrow", ...). Phrases are stored
    /// pre-split so matching them costs only span comparisons against the already-lexed words.
    /// </summary>
    public sealed class Phrase
    {
        public readonly string[] Words;
        public readonly TermInfo Info;

        public Phrase(string[] words, TermInfo info)
        {
            Words = words;
            Info  = info;
        }
    }

    /// <summary>
    /// A language's vocabulary. Single words are resolved through a <see cref="FrozenDictionary{TKey,TValue}"/>
    /// alternate lookup, so a <c>ReadOnlySpan&lt;char&gt;</c> is matched without ever materialising a string.
    /// Phrases are bucketed by their first word for the same reason.
    /// </summary>
    public sealed class Lexicon
    {
        private readonly FrozenDictionary<string, TermInfo>                                              _words;
        private readonly FrozenDictionary<string, TermInfo>.AlternateLookup<ReadOnlySpan<char>>          _wordsBySpan;
        private readonly FrozenSet<string>                                                               _suffixes;
        private readonly FrozenSet<string>.AlternateLookup<ReadOnlySpan<char>>                           _suffixesBySpan;
        private readonly FrozenDictionary<string, Phrase[]>                                              _phrases;
        private readonly FrozenDictionary<string, Phrase[]>.AlternateLookup<ReadOnlySpan<char>>          _phrasesBySpan;

        public Language Language      { get; }
        public bool     DayMonthOrder { get; }
        /// <summary>True where a fraction is written with a comma ("123,45 sec").</summary>
        public bool     DecimalComma  { get; }
        /// <summary>
        /// Whether a leading definite article belongs to a date's span. English keeps it ("the 09th of may");
        /// every other language reports the date without it ("le 4 janvier 2019" is "4 janvier 2019").
        /// </summary>
        public bool     ArticleInDateSpan { get; }

        /// <summary>
        /// Whether a leading definite article belongs to a qualified period's span ("la semaine prochaine").
        /// Most languages drop it, the way English reports "the april 2017" as "april 2017".
        /// </summary>
        public bool     ArticleInPeriodSpan { get; }

        /// <summary>
        /// Whether the qualifier may follow the unit ("la semaine prochaine"). English puts it in front, so
        /// reading it the other way round turns "2 hours next month" into a two-hour period.
        /// </summary>
        public bool     RelativeAfterUnit { get; }

        /// <summary>
        /// Whether a plural unit can be told from a singular one by its last letter. Where it can,
        /// "3 next week" is not a period of three weeks — it is the number three beside "next week".
        /// </summary>
        public bool     PluralEndsInS { get; }

        /// <summary>
        /// Whether naming part of a period needs a preposition ("the end of may"). Where it does, a bare
        /// "start" or "end" in front of something else is the verb; where it does not ("Anfang Mai"), it is not.
        /// </summary>
        public bool     PartNamedWithOf { get; }

        /// <summary>
        /// Whether the minutes are spoken after the hour and joined to it ("siete y media" is half past
        /// seven). English and the Germanic languages say it the other way round.
        /// </summary>
        public bool     MinutesFollowHour { get; }

        /// <summary>
        /// Whether the language writes compounds as one word ("dienstagmorgen", "maandagmiddag"), so an
        /// unknown word is worth splitting into two the lexicon does know.
        /// </summary>
        public bool     SplitsCompounds { get; }

        /// <summary>
        /// Whether "half" names the half hour before the hour it precedes: "halb acht" is half past seven.
        /// </summary>
        public bool     HalfIsBeforeTheHour { get; }

        /// <summary>
        /// Whether an ordinal is written as its number and a full stop: German's "22. April".
        /// </summary>
        public bool     OrdinalEndsInDot { get; }

        /// <summary>
        /// Whether the hour unit standing after a number names the time of day: Portuguese "3 horas" and
        /// French "3 heures" are three o'clock, where Spanish "3 horas" and Italian "3 ore" are three hours
        /// long. A preposition in front of it still opens a length — "por 3 horas", "pour 3 heures".
        /// </summary>
        public bool     HourUnitNamesTheClock { get; }

        /// <summary>
        /// Whether a holiday that falls on a different day each year still names a day in its timex.
        /// The suites disagree: English reports "easter monday" as XXXX-04-22, German reports
        /// "Ostermontag" as XXXX. A fixed holiday always names its day, whatever this says.
        /// </summary>
        public bool     MovableHolidayNamesItsDay { get; }

        /// <summary>
        /// Whether a narrowing word may follow the period it narrows: German's "dieses Jahr früh" is the
        /// first half of this year. English writes it in front, so a trailing "early" there is a new phrase.
        /// </summary>
        public bool     NarrowingFollowsPeriod { get; }

        public Lexicon(Language language, bool dayMonthOrder, IEnumerable<KeyValuePair<string, TermInfo>> words, IEnumerable<KeyValuePair<string, TermInfo>> phrases, bool decimalComma = false, bool articleInDateSpan = true, bool articleInPeriodSpan = false, bool relativeAfterUnit = false, bool pluralEndsInS = true, bool partNamedWithOf = false, bool minutesFollowHour = false, bool splitsCompounds = false, bool halfIsBeforeTheHour = false, bool ordinalEndsInDot = false, bool movableHolidayNamesItsDay = true, bool hourUnitNamesTheClock = false, bool narrowingFollowsPeriod = false)
        {
            NarrowingFollowsPeriod    = narrowingFollowsPeriod;
            HourUnitNamesTheClock     = hourUnitNamesTheClock;
            MovableHolidayNamesItsDay = movableHolidayNamesItsDay;
            OrdinalEndsInDot  = ordinalEndsInDot;
            PartNamedWithOf   = partNamedWithOf;
            MinutesFollowHour = minutesFollowHour;
            SplitsCompounds   = splitsCompounds;
            HalfIsBeforeTheHour = halfIsBeforeTheHour;
            Language      = language;
            DayMonthOrder = dayMonthOrder;
            DecimalComma  = decimalComma;
            ArticleInDateSpan    = articleInDateSpan;
            ArticleInPeriodSpan  = articleInPeriodSpan;
            RelativeAfterUnit    = relativeAfterUnit;
            PluralEndsInS        = pluralEndsInS;

            var singles = new Dictionary<string, TermInfo>(StringComparer.OrdinalIgnoreCase);
            var multi   = new Dictionary<string, List<Phrase>>(StringComparer.OrdinalIgnoreCase);
            var suffixes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var kv in words)
            {
                // An ordinal suffix is a letter the language also uses for something else — Portuguese's
                // "o" is its article and "a" its preposition — so it is remembered apart from the reading
                // that ends up winning the word
                if (kv.Value.Is(TermKind.OrdinalSuffix)) suffixes.Add(kv.Key);

                singles[kv.Key] = kv.Value;
            }

            foreach (var kv in phrases)
            {
                var parts = kv.Key.Split(' ', StringSplitOptions.RemoveEmptyEntries);

                if (parts.Length == 1)
                {
                    singles[parts[0]] = kv.Value;
                    continue;
                }

                if (!multi.TryGetValue(parts[0], out var list))
                {
                    list = new List<Phrase>();
                    multi[parts[0]] = list;
                }

                list.Add(new Phrase(parts, kv.Value));
            }

            var byFirstWord = new Dictionary<string, Phrase[]>(StringComparer.OrdinalIgnoreCase);

            foreach (var kv in multi)
            {
                // Longest first, so "new year's eve" wins over "new year"
                kv.Value.Sort(static (a, b) => b.Words.Length.CompareTo(a.Words.Length));
                byFirstWord[kv.Key] = kv.Value.ToArray();
            }

            _words         = singles.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);
            _wordsBySpan   = _words.GetAlternateLookup<ReadOnlySpan<char>>();
            _phrases       = byFirstWord.ToFrozenDictionary(StringComparer.OrdinalIgnoreCase);
            _phrasesBySpan = _phrases.GetAlternateLookup<ReadOnlySpan<char>>();
            _suffixes      = suffixes.ToFrozenSet(StringComparer.OrdinalIgnoreCase);
            _suffixesBySpan = _suffixes.GetAlternateLookup<ReadOnlySpan<char>>();
        }

        /// <summary>
        /// Splits a word the lexicon does not know into two it does, where the language writes compounds as
        /// one word: "dienstagmorgen" is tuesday plus morning. Both halves have to name something.
        /// </summary>
        public bool TrySplitCompound(ReadOnlySpan<char> word, out int cut, out TermInfo head, out TermInfo tail)
        {
            cut  = 0;
            head = default;
            tail = default;

            if (!SplitsCompounds || word.Length < 8) return false;

            // The longest head that leaves a word behind: "montagnachmittag" is montag, not mona
            for (int k = word.Length - 3; k >= 3; k--)
            {
                if (!_wordsBySpan.TryGetValue(word[..k], out head))  continue;
                if (!_wordsBySpan.TryGetValue(word[k..], out tail))  continue;
                // "spätabends" qualifies its tail, "zweieinhalb" counts its fraction; every other
                // compound is two things written as one
                bool fraction = head.Kind == TermKind.Cardinal && tail.Kind is TermKind.HalfWord or TermKind.QuarterWord;

                if (!NamesSomething(head) && head.Kind != TermKind.Mod && !fraction) continue;
                if (!NamesSomething(tail) && !fraction)                              continue;

                cut = k;
                return true;
            }

            return false;
        }

        /// <summary>
        /// A number the language writes as one word from its units and tens: "neunundzwanzig" is
        /// nine-and-twenty. The joiner is the language's "and", which is a Connector in the lexicon.
        /// </summary>
        /// <summary>
        /// "un'ora", "l'altro" — a word elided onto the next one. The apostrophe belongs to the first
        /// half, which is how the lexicon spells it ("l'", "un'"), and both halves must be known words.
        /// </summary>
        public bool TrySplitElision(ReadOnlySpan<char> word, out int cut, out TermInfo head, out TermInfo tail)
        {
            cut = 0; head = default; tail = default;

            int mark = -1;
            for (int k = 1; k < word.Length - 1; k++)
            {
                if (word[k] is '\'' or '\u2019' or 'ʼ') { mark = k; break; }
            }

            if (mark < 0) return false;

            if (!_wordsBySpan.TryGetValue(word[..(mark + 1)], out head)) return false;
            if (!_wordsBySpan.TryGetValue(word[(mark + 1)..], out tail)) return false;

            cut = mark + 1;
            return true;
        }

        public bool TrySplitNumber(ReadOnlySpan<char> word, out TermInfo number)
        {
            number = default;

            if (word.Length < 7) return false;

            // "ventinove", "veinticuatro" — the tens and the units run together, with nothing between them
            for (int k = 4; k <= word.Length - 3; k++)
            {
                if (!_wordsBySpan.TryGetValue(word[..k], out var leadTens))                                  continue;
                if (leadTens.Kind != TermKind.Cardinal)                                                      continue;
                if (leadTens.Value < 20 || leadTens.Value > 90 || leadTens.Value % 10 != 0)                   continue;
                if (!_wordsBySpan.TryGetValue(word[k..], out var trailUnits))                                 continue;
                if (trailUnits.Kind is not TermKind.Cardinal and not TermKind.Ordinal)                        continue;
                if (trailUnits.Value < 1 || trailUnits.Value > 9)                                             continue;

                number = new TermInfo(trailUnits.Kind, leadTens.Value + trailUnits.Value);
                return true;
            }

            if (!SplitsCompounds || word.Length < 8) return false;

            for (int k = 3; k <= word.Length - 5; k++)
            {
                if (!_wordsBySpan.TryGetValue(word[..k], out var units)) continue;
                if (units.Kind != TermKind.Cardinal || units.Value < 1 || units.Value > 9) continue;

                for (int j = k + 2; j <= k + 3 && j <= word.Length - 4; j++)
                {
                    if (!_wordsBySpan.TryGetValue(word[k..j], out var joiner) || joiner.Kind != TermKind.Connector) continue;
                    if (!_wordsBySpan.TryGetValue(word[j..], out var tens)) continue;
                    if (tens.Value < 20 || tens.Value > 90 || tens.Value % 10 != 0) continue;
                    if (tens.Kind != TermKind.Cardinal && tens.Kind != TermKind.Ordinal) continue;

                    number = new TermInfo(tens.Kind, tens.Value + units.Value);
                    return true;
                }
            }

            return false;
        }

        private static bool NamesSomething(TermInfo info) => NamesSomething(info.Kind) || NamesSomething(info.AltKind);

        private static bool NamesSomething(TermKind kind) => kind is TermKind.Weekday or TermKind.Month
            or TermKind.SpecialDay or TermKind.PartOfDay or TermKind.Unit or TermKind.Relative or TermKind.Season
            or TermKind.HalfWord or TermKind.QuarterWord;

        /// <summary>
        /// Whether the word reads as the plural of another word this lexicon knows. Spanish and French name
        /// most of their weekdays with a final -s that is not one — "martes" is a Tuesday, not Tuesdays —
        /// so the letter alone cannot say, but "marte" being no word of theirs can.
        /// </summary>
        public bool IsPluralOfKnownWord(ReadOnlySpan<char> word)
        {
            if (word.Length < 2 || (word[^1] != 's' && word[^1] != 'S')) return false;

            return _wordsBySpan.ContainsKey(word[..^1]) || (word.Length > 2 && _wordsBySpan.ContainsKey(word[..^2]));
        }

        public bool TryGetWord(ReadOnlySpan<char> word, out TermInfo info)
        {
            if (_wordsBySpan.TryGetValue(word, out info)) return true;

            // "this week's" is the same word as "this week"
            if (word.Length > 2 && (word[^1] == 's' || word[^1] == 'S') && (word[^2] == '\'' || word[^2] == '\u2019'))
            {
                return _wordsBySpan.TryGetValue(word[..^2], out info);
            }

            return false;
        }

        public bool TryGetPhrases(ReadOnlySpan<char> firstWord, out Phrase[] phrases) => _phrasesBySpan.TryGetValue(firstWord, out phrases);

        /// <summary>Whether the word is one the language writes an ordinal's number with ("9º", "9o", "6a").</summary>
        public bool IsOrdinalSuffix(ReadOnlySpan<char> word) => _suffixesBySpan.Contains(word);
    }
}
