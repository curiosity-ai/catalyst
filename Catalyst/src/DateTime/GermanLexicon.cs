using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The German vocabulary.</summary>
    public static class GermanLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "januar", "jänner", "jan", "jan.");
            b.Add(TermKind.Month,  2, "februar", "feb", "febr");
            b.Add(TermKind.Month,  3, "märz", "maerz", "mrz", "mär");
            b.Add(TermKind.Month,  4, "april", "apr");
            b.Add(TermKind.Month,  5, "mai");
            b.Add(TermKind.Month,  6, "juni", "jun");
            b.Add(TermKind.Month,  7, "juli", "jul");
            b.Add(TermKind.Month,  8, "august", "aug");
            b.Add(TermKind.Month,  9, "september", "sep", "sept");
            b.Add(TermKind.Month, 10, "oktober", "okt");
            b.Add(TermKind.Month, 11, "november", "nov");
            b.Add(TermKind.Month, 12, "dezember", "dez");

            b.Add(TermKind.Weekday, 0, "sonntag", "sonntags", "so");
            b.Add(TermKind.Weekday, 1, "montag", "montags", "mo");
            b.Add(TermKind.Weekday, 2, "dienstag", "dienstags", "di");
            b.Add(TermKind.Weekday, 3, "mittwoch", "mittwochs", "mi");
            b.Add(TermKind.Weekday, 4, "donnerstag", "donnerstags", "do");
            b.Add(TermKind.Weekday, 5, "freitag", "freitags", "fr");
            b.Add(TermKind.Weekday, 6, "samstag", "samstags", "sonnabend", "sa");

            b.Add(TermKind.Cardinal,  0, "null");
            b.Add(TermKind.Cardinal,  1, "ein", "eine", "einen", "einer", "eins");
            b.Add(TermKind.Cardinal,  2, "zwei", "zwo");
            b.Add(TermKind.Cardinal,  3, "drei");
            b.Add(TermKind.Cardinal,  4, "vier");
            b.Add(TermKind.Cardinal,  5, "fünf", "fuenf");
            b.Add(TermKind.Cardinal,  6, "sechs");
            b.Add(TermKind.Cardinal,  7, "sieben");
            b.Add(TermKind.Cardinal,  8, "acht");
            b.Add(TermKind.Cardinal,  9, "neun");
            b.Add(TermKind.Cardinal, 10, "zehn");
            b.Add(TermKind.Cardinal, 11, "elf");
            b.Add(TermKind.Cardinal, 12, "zwölf", "zwoelf");
            b.Add(TermKind.Cardinal, 13, "dreizehn");
            b.Add(TermKind.Cardinal, 14, "vierzehn");
            b.Add(TermKind.Cardinal, 15, "fünfzehn", "fuenfzehn");
            b.Add(TermKind.Cardinal, 16, "sechzehn");
            b.Add(TermKind.Cardinal, 17, "siebzehn");
            b.Add(TermKind.Cardinal, 18, "achtzehn");
            b.Add(TermKind.Cardinal, 19, "neunzehn");
            b.Add(TermKind.Cardinal, 20, "zwanzig");
            b.Add(TermKind.Cardinal, 30, "dreißig", "dreissig");
            b.Add(TermKind.Cardinal, 40, "vierzig");
            b.Add(TermKind.Cardinal, 50, "fünfzig", "fuenfzig");
            b.Add(TermKind.Cardinal, 60, "sechzig");
            b.Add(TermKind.Cardinal, 70, "siebzig");
            b.Add(TermKind.Cardinal, 80, "achtzig");
            b.Add(TermKind.Cardinal, 90, "neunzig");
            b.Add(TermKind.Multiplier,    100, "hundert");
            b.Add(TermKind.Multiplier,   1000, "tausend");
            b.Add(TermKind.Multiplier, 1000000, "million", "millionen");

            b.Add(TermKind.Ordinal,  1, "erste", "erster", "ersten", "erstes");
            b.Add(TermKind.Ordinal,  2, "zweite", "zweiter", "zweiten", "zweites");
            b.Add(TermKind.Ordinal,  3, "dritte", "dritter", "dritten", "drittes");
            b.Add(TermKind.Ordinal,  4, "vierte", "vierter", "vierten", "viertes");
            b.Add(TermKind.Ordinal,  5, "fünfte", "fuenfte", "fünfter", "fünften");
            b.Add(TermKind.Ordinal,  6, "sechste", "sechster", "sechsten");
            b.Add(TermKind.Ordinal,  7, "siebte", "siebter", "siebten", "siebente");
            b.Add(TermKind.Ordinal,  8, "achte", "achter", "achten");
            b.Add(TermKind.Ordinal,  9, "neunte", "neunter", "neunten");
            b.Add(TermKind.Ordinal, 10, "zehnte", "zehnter", "zehnten");
            b.Add(TermKind.Ordinal, 11, "elfte", "elfter", "elften");
            b.Add(TermKind.Ordinal, 12, "zwölfte", "zwoelfte");
            b.Add(TermKind.Ordinal, 13, "dreizehnte");
            b.Add(TermKind.Ordinal, 20, "zwanzigste");
            b.Add(TermKind.Ordinal, 30, "dreißigste", "dreissigste");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,    "sekunde", "sekunden", "sek");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,    "minute", "minuten", "min");
            b.Add(TermKind.Unit, (int)TimeUnit.Hour,      "stunde", "stunden", "std", "h");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,       "tag", "tage", "tagen", "tages", "t");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,      "woche", "wochen", "wo");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "vierzehn tage");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,     "monat", "monate", "monaten", "monats");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter,   "quartal", "quartale", "vierteljahr");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,      "jahr", "jahre", "jahren", "jahres", "j");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,    "jahrzehnt", "jahrzehnte", "dekade");
            b.Add(TermKind.Unit, (int)TimeUnit.Century,   "jahrhundert", "jahrhunderte");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend,   "wochenende", "wochenenden");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "nacht", "nächte", "naechte");
            b.Add(TermKind.BusinessDay, "arbeits", "werk", "geschäfts", "werktag", "werktage", "arbeitstag", "arbeitstage");
            b.Add(TermKind.Several, 3, "einige", "mehrere", "ein paar", "manche");
            b.Add(TermKind.Several, 2, "paar");
            b.Add(TermKind.HalfWord, "halbe", "halb", "halben");
            b.Add(TermKind.QuarterWord, "viertel");

            b.Add(TermKind.Relative, (int)RelativeKind.This,      "diese", "dieser", "diesen", "dieses", "diesem", "kommende", "kommenden");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,      "nächste", "naechste", "nächsten", "naechsten", "nächster", "nächstes", "folgende", "folgenden");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,      "letzte", "letzten", "letzter", "letztes", "vergangene", "vergangenen", "vorige", "vorigen");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous,  "vorherige", "vorherigen", "vorletzte");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,   "aktuelle", "aktuellen", "laufende", "laufenden", "selbe", "selben", "gleiche", "gleichen");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "heute");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow,           "morgen");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "gestern");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "übermorgen", "uebermorgen");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "vorgestern");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "jetzt", "sofort", "gerade", "im moment", "momentan");

            b.Add(TermKind.Ago,     "vor", "davor", "früher", "frueher");
            b.Add(TermKind.FromNow, "später", "spaeter", "danach", "ab jetzt");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "vormittag", "vormittags", "morgens", "früh", "frueh");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "nachmittag", "nachmittags");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "abend", "abends");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "mittag", "mittags");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "mitternacht");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "heute abend", "heute nacht");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "mittagessen", "mittagszeit");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "abendessen", "abendbrot");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "frühstück", "fruehstueck");

            b.Add(TermKind.AmPm, 0, "vormittags", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "nachmittags", "pm", "p.m.");
            b.Add(TermKind.OClock, "uhr");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "bis", "zu", "zum", "bis zum", "bis zu");
            b.Add(TermKind.Connector, "und");
            b.Add(TermKind.RangeStart, 0, "von", "vom", "ab", "seit", "beginnend");
            b.Add(TermKind.RangeStart, 1, "zwischen");

            b.Add(TermKind.Filler, "der", "die", "das", "den", "dem", "des", "am", "im", "in", "an", "auf", "um", "für", "fuer", "einem", "einer", "eines");

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "vor dem", "bis spätestens", "spätestens");
            b.Add(TermKind.Mod, (int)ModKind.After,     "nach", "nach dem", "später als");
            b.Add(TermKind.Mod, (int)ModKind.Less,      "weniger als");
            b.Add(TermKind.Mod, (int)ModKind.More,      "mehr als");
            b.Add(TermKind.Mod, (int)ModKind.Since,     "seit dem");
            b.Add(TermKind.Mod, (int)ModKind.Start,     "anfang", "beginn", "anfang des", "zu beginn");
            b.Add(TermKind.Mod, (int)ModKind.End,       "ende", "ende des", "zum ende");
            b.Add(TermKind.Mod, (int)ModKind.Mid,       "mitte", "mitte des");
            b.Add(TermKind.Mod, (int)ModKind.Early,     "früh im", "anfangs");
            b.Add(TermKind.Mod, (int)ModKind.Late,      "spät im", "ende von");
            b.Add(TermKind.Approx, "gegen", "etwa", "ungefähr", "ungefaehr", "circa", "ca");

            b.Add(TermKind.SetPrefix, 0, "jeden", "jede", "jedes", "jedem", "alle");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "täglich", "taeglich");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "wöchentlich", "woechentlich");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "monatlich");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "jährlich", "jaehrlich");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "stündlich", "stuendlich");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "vierteljährlich");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "frühling", "fruehling", "frühjahr");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "sommer");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "herbst");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "winter");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,        "neujahr", "neujahrstag");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,     "silvester");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,      "weihnachten", "weihnachtstag", "erster weihnachtstag");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve,   "heiligabend", "heiliger abend");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,         "ostern", "ostersonntag");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday,   "ostermontag");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,     "karfreitag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Boxing,         "zweiter weihnachtstag", "stephanstag");
            b.Add(TermKind.Holiday, (int)HolidayKind.GermanUnityDay, "tag der deutschen einheit");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "tag der arbeit");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,      "allerheiligen");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,     "valentinstag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,      "halloween");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,     "muttertag");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,     "vatertag");

            return b.Build(Language.German, dayMonthOrder: true, decimalComma: true, pluralEndsInS: false);
        }
    }
}
