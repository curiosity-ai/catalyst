using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The Dutch vocabulary.</summary>
    public static class DutchLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "januari", "jan");
            b.Add(TermKind.Month,  2, "februari", "feb");
            b.Add(TermKind.Month,  3, "maart", "mrt", "maa");
            b.Add(TermKind.Month,  4, "april", "apr");
            b.Add(TermKind.Month,  5, "mei");
            b.Add(TermKind.Month,  6, "juni", "jun");
            b.Add(TermKind.Month,  7, "juli", "jul");
            b.Add(TermKind.Month,  8, "augustus", "aug");
            b.Add(TermKind.Month,  9, "september", "sep", "sept");
            b.Add(TermKind.Month, 10, "oktober", "okt");
            b.Add(TermKind.Month, 11, "november", "nov");
            b.Add(TermKind.Month, 12, "december", "dec");

            b.Add(TermKind.Weekday, 0, "zondag", "zondagen", "zo");
            b.Add(TermKind.Weekday, 1, "maandag", "maandagen", "ma");
            b.Add(TermKind.Weekday, 2, "dinsdag", "dinsdagen", "di");
            b.Add(TermKind.Weekday, 3, "woensdag", "woensdagen", "wo");
            b.Add(TermKind.Weekday, 4, "donderdag", "donderdagen", "do");
            b.Add(TermKind.Weekday, 5, "vrijdag", "vrijdagen", "vr");
            b.Add(TermKind.Weekday, 6, "zaterdag", "zaterdagen", "za");

            b.Add(TermKind.Cardinal,  0, "nul");
            b.Add(TermKind.Cardinal,  1, "één");
            b.Add(TermKind.Cardinal,  2, "twee");
            b.Add(TermKind.Cardinal,  3, "drie");
            b.Add(TermKind.Cardinal,  4, "vier");
            b.Add(TermKind.Cardinal,  5, "vijf");
            b.Add(TermKind.Cardinal,  6, "zes");
            b.Add(TermKind.Cardinal,  7, "zeven");
            b.Add(TermKind.Cardinal,  8, "acht");
            b.Add(TermKind.Cardinal,  9, "negen");
            b.Add(TermKind.Cardinal, 10, "tien");
            b.Add(TermKind.Cardinal, 11, "elf");
            b.Add(TermKind.Cardinal, 12, "twaalf");
            b.Add(TermKind.Cardinal, 13, "dertien");
            b.Add(TermKind.Cardinal, 14, "veertien");
            b.Add(TermKind.Cardinal, 15, "vijftien");
            b.Add(TermKind.Cardinal, 16, "zestien");
            b.Add(TermKind.Cardinal, 17, "zeventien");
            b.Add(TermKind.Cardinal, 18, "achttien");
            b.Add(TermKind.Cardinal, 19, "negentien");
            b.Add(TermKind.Cardinal, 20, "twintig");
            b.Add(TermKind.Cardinal, 30, "dertig");
            b.Add(TermKind.Cardinal, 40, "veertig");
            b.Add(TermKind.Cardinal, 50, "vijftig");
            b.Add(TermKind.Cardinal, 60, "zestig");
            b.Add(TermKind.Cardinal, 70, "zeventig");
            b.Add(TermKind.Cardinal, 80, "tachtig");
            b.Add(TermKind.Cardinal, 90, "negentig");
            b.Add(TermKind.Multiplier,     100, "honderd");
            b.Add(TermKind.Multiplier,    1000, "duizend");
            b.Add(TermKind.Multiplier, 1000000, "miljoen");

            b.Add(TermKind.Ordinal,  1, "eerste");
            b.Add(TermKind.Ordinal,  2, "tweede");
            b.Add(TermKind.Ordinal,  3, "derde");
            b.Add(TermKind.Ordinal,  4, "vierde");
            b.Add(TermKind.Ordinal,  5, "vijfde");
            b.Add(TermKind.Ordinal,  6, "zesde");
            b.Add(TermKind.Ordinal,  7, "zevende");
            b.Add(TermKind.Ordinal,  8, "achtste");
            b.Add(TermKind.Ordinal,  9, "negende");
            b.Add(TermKind.Ordinal, 10, "tiende");
            b.Add(TermKind.Ordinal, 20, "twintigste");
            b.Add(TermKind.Ordinal, 30, "dertigste");
            b.Add(TermKind.OrdinalSuffix, "e", "de", "ste");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,  "seconde", "seconden", "sec");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,  "minuut", "minuten", "min");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "uur", "uren", "u");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,     "dag", "dagen", "d");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,    "week", "weken", "wk");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,   "maand", "maanden");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter, "kwartaal", "kwartalen");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,    "jaar", "jaren", "jr");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,  "decennium", "decennia");
            b.Add(TermKind.Unit, (int)TimeUnit.Century, "eeuw", "eeuwen");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend, "weekend", "weekenden");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "nacht", "nachten");
            b.Add(TermKind.BusinessDay, "werk", "werkdag", "werkdagen");
            b.Add(TermKind.Several, 3, "enkele", "enige", "wat", "een paar", "verscheidene", "sommige", "meerdere");
            b.Add(TermKind.Several, 2, "paar");
            b.Add(TermKind.HalfWord, "halve");
            b.Add(TermKind.QuarterWord, "kwartier", "kwart");

            b.Add(TermKind.Relative, (int)RelativeKind.This,     "deze", "dit");
            b.Add(TermKind.Relative, (int)RelativeKind.Coming,   "komende", "komend", "aanstaande", "aanstaand");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,     "volgende", "volgend", "aankomende");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,     "vorige", "vorig", "afgelopen", "laatste");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous, "voorgaande", "voorafgaande");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,  "huidige", "huidig", "zelfde", "dezelfde", "dat", "die");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "vandaag");
            b.Add(new TermInfo(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow, TermKind.PartOfDay, (int)PartOfDayKind.Morning), "morgen");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "gisteren");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "overmorgen");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "eergisteren");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "nu", "op dit moment", "meteen");

            b.Add(TermKind.Ago,     "geleden");
            b.Add(new TermInfo(TermKind.Ago, 0, TermKind.Mod, (int)ModKind.Earlier), "eerder");
            b.Add(TermKind.FromNow, "vanaf nu", "daarna");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later), "later");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "ochtend", "ochtenden", "'s ochtends", "'s morgens", "vanmorgen", "vanochtend", "voormiddag");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "middag", "namiddag", "'s middags", "vanmiddag");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "avond", "avonden", "'s avonds");
            b.Add(new TermInfo(TermKind.PastWord, 0, TermKind.InPrefix, 0), "over");
            // "voor 2010" is before 2010, and "10 voor half negen" counts off the hour
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "voor");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "middaguur", "twaalf uur 's middags");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "middernacht");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "vanavond", "vannacht");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "lunch", "lunchtijd");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "diner", "avondeten");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "ontbijt");

            b.Add(TermKind.AmPm, 0, "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "pm", "p.m.");
            b.Add(TermKind.OClock, "uur precies");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "tot", "tot en met", "t/m", "naar");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.AndWord), "en");
            b.Add(TermKind.RangeStart, 0, "vanaf", "beginnend");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.RangeStart, 0), "van");   // "de 9e van mei" and "van 4-23"
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Since, TermKind.RangeStart, 0), "sinds");
            b.Add(TermKind.RangeStart, 1, "tussen");

            b.Add(TermKind.Filler, "in", "op", "om", "van de", "aan");
            b.Add(new TermInfo(TermKind.Article, 0, TermKind.Filler, 0), "het");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.OrdinalSuffix), "de");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.Cardinal, 1), "een");
            b.Add(new TermInfo(TermKind.Whole, 0, TermKind.Filler, 0), "hele", "heel", "gehele", "geheel");   // "the whole day" counts as one
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.InPrefix, 0), "in");
                        b.Add(TermKind.InPrefix, 1, "binnen");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "om");

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "voor de", "uiterlijk", "niet later dan");
            b.Add(TermKind.Mod, (int)ModKind.After, "na", "na de", "later dan", "hoger dan", "groter dan");
            b.Add(TermKind.Mod, (int)ModKind.Less,  "minder dan");
            b.Add(TermKind.Mod, (int)ModKind.More,  "meer dan");
            b.Add(TermKind.Mod, (int)ModKind.Start, "begin", "begin van", "start van",
                                                    "beginnend", "beginnende", "startend", "startende", "aanvangend", "aanvangende");
            b.Add(TermKind.Mod, (int)ModKind.End,   "eind", "einde", "eind van", "einde van");
            b.Add(TermKind.Mod, (int)ModKind.Mid,   "midden", "midden van", "halverwege");
            b.Add(TermKind.Mod, (int)ModKind.Since,  "sindsdien", "zodra");
            b.Add(TermKind.Mod, (int)ModKind.Early,  "vroeg in", "vroeg");
            b.Add(TermKind.Mod, (int)ModKind.Late,   "laat in", "laat");
            b.Add(TermKind.Mod, (int)ModKind.OrLater,   "of later", "en later", "of daarna", "en daarna", "of hoger", "en hoger");
            b.Add(TermKind.Mod, (int)ModKind.OrEarlier, "of eerder", "en eerder", "of ervoor", "en ervoor");
            b.Add(new TermInfo(TermKind.HalfWord, 0, TermKind.Mod, (int)ModKind.Mid), "half");   // "half acht" and "half augustus"
            b.Add(TermKind.Approx, "rond", "omstreeks", "ongeveer", "circa");

            b.Add(TermKind.SetPrefix, 0, "elke", "elk", "iedere", "ieder", "alle");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "dagelijks");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "wekelijks");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "maandelijks");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "jaarlijks");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "elk uur");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "per kwartaal");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "lente", "voorjaar");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "zomer");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "herfst", "najaar");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "winter");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,      "nieuwjaar", "nieuwjaarsdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,   "oudejaarsavond", "oud en nieuw");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,    "kerst", "kerstmis", "eerste kerstdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve, "kerstavond");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,       "pasen", "eerste paasdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday, "tweede paasdag", "paasmaandag");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,   "goede vrijdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Boxing,       "tweede kerstdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,    "allerheiligen");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,   "valentijnsdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,    "halloween");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,   "moederdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,   "vaderdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "dag van de arbeid");
            b.Add(TermKind.Holiday, (int)HolidayKind.KingsDay,     "koningsdag", "koninginnedag");
            b.Add(TermKind.Holiday, (int)HolidayKind.IndependenceDay, "onafhankelijkheidsdag", "onafhankelijksdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.BlackFriday,  "black friday", "zwarte vrijdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.CyberMonday,  "cyber monday");
            b.Add(TermKind.Holiday, (int)HolidayKind.EarthDay,     "dag van de aarde", "wereldaarddag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Juneteenth,   "juneteenth");
            b.Add(TermKind.Holiday, (int)HolidayKind.FreedomDay,   "vrijheidsdag", "bevrijdingsdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.JubileeDay,   "jubilee day", "jubileumdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.StPatricksDay, "sint-patricksdag", "st patricksdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Thanksgiving, "thanksgiving", "dankdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Pentecost,    "pinksteren", "eerste pinksterdag");
            b.Add(TermKind.Holiday, (int)HolidayKind.PalmSunday,   "palmzondag");
            b.Add(TermKind.Holiday, (int)HolidayKind.Epiphany,     "driekoningen");

            return b.Build(Language.Dutch, dayMonthOrder: true, decimalComma: true, articleInDateSpan: false, articleInPeriodSpan: false, pluralEndsInS: false, splitsCompounds: true, halfIsBeforeTheHour: true, boundsOnlyOnTimes: true);
        }
    }
}
