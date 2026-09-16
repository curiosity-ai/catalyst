using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The Italian vocabulary.</summary>
    public static class ItalianLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "gennaio", "gen");
            b.Add(TermKind.Month,  2, "febbraio", "feb");
            b.Add(TermKind.Month,  3, "marzo", "mar");
            b.Add(TermKind.Month,  4, "aprile", "apr");
            b.Add(TermKind.Month,  5, "maggio", "mag");
            b.Add(TermKind.Month,  6, "giugno", "giu");
            b.Add(TermKind.Month,  7, "luglio", "lug");
            b.Add(TermKind.Month,  8, "agosto", "ago");
            b.Add(TermKind.Month,  9, "settembre", "set");
            b.Add(TermKind.Month, 10, "ottobre", "ott");
            b.Add(TermKind.Month, 11, "novembre", "nov");
            b.Add(TermKind.Month, 12, "dicembre", "dic");

            b.Add(TermKind.Weekday, 0, "domenica", "domeniche", "dom");
            b.Add(TermKind.Weekday, 1, "lunedì", "lunedi", "lun");
            b.Add(TermKind.Weekday, 2, "martedì", "martedi", "mar");
            b.Add(TermKind.Weekday, 3, "mercoledì", "mercoledi", "mer");
            b.Add(TermKind.Weekday, 4, "giovedì", "giovedi", "gio");
            b.Add(TermKind.Weekday, 5, "venerdì", "venerdi", "ven");
            b.Add(TermKind.Weekday, 6, "sabato", "sabati", "sab");

            b.Add(TermKind.Cardinal,  0, "zero");
            b.Add(TermKind.Cardinal,  1, "uno", "una", "un");
            b.Add(TermKind.Cardinal,  2, "due");
            b.Add(TermKind.Cardinal,  3, "tre");
            b.Add(TermKind.Cardinal,  4, "quattro");
            b.Add(TermKind.Cardinal,  5, "cinque");
            b.Add(TermKind.Cardinal,  6, "sei");
            b.Add(TermKind.Cardinal,  7, "sette");
            b.Add(TermKind.Cardinal,  8, "otto");
            b.Add(TermKind.Cardinal,  9, "nove");
            b.Add(TermKind.Cardinal, 10, "dieci");
            b.Add(TermKind.Cardinal, 11, "undici");
            b.Add(TermKind.Cardinal, 12, "dodici");
            b.Add(TermKind.Cardinal, 13, "tredici");
            b.Add(TermKind.Cardinal, 14, "quattordici");
            b.Add(TermKind.Cardinal, 15, "quindici");
            b.Add(TermKind.Cardinal, 16, "sedici");
            b.Add(TermKind.Cardinal, 17, "diciassette");
            b.Add(TermKind.Cardinal, 18, "diciotto");
            b.Add(TermKind.Cardinal, 19, "diciannove");
            b.Add(TermKind.Cardinal, 20, "venti", "vent");   // "ventuno" elides the tens
            b.Add(TermKind.Cardinal, 30, "trenta");
            b.Add(TermKind.Cardinal, 40, "quaranta");
            b.Add(TermKind.Cardinal, 50, "cinquanta");
            b.Add(TermKind.Cardinal, 60, "sessanta");
            b.Add(TermKind.Cardinal, 70, "settanta");
            b.Add(TermKind.Cardinal, 80, "ottanta");
            b.Add(TermKind.Cardinal, 90, "novanta");
            b.Add(TermKind.Multiplier,     100, "cento");
            b.Add(TermKind.Multiplier,    1000, "mille", "mila");
            b.Add(TermKind.Multiplier, 1000000, "milione", "milioni");

            b.Add(TermKind.Ordinal,  1, "primo");
            b.Add(TermKind.Ordinal,  2, "seconda");
            b.Add(TermKind.Ordinal,  3, "terzo", "terza");
            b.Add(TermKind.Ordinal,  4, "quarta");
            b.Add(TermKind.Ordinal,  5, "quinto", "quinta");
            b.Add(TermKind.Ordinal,  6, "sesto");
            b.Add(TermKind.Ordinal,  7, "settimo");
            b.Add(TermKind.Ordinal,  8, "ottavo");
            b.Add(TermKind.Ordinal,  9, "nono");
            b.Add(TermKind.Ordinal, 10, "decimo");
            b.Add(TermKind.OrdinalSuffix, "º", "ª", "°");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,  "secondi", "sec");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Second, TermKind.Ordinal, 2), "secondo");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,  "minuto", "minuti", "min");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "ora", "ore", "h");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,     "giorno", "giorni", "giornata", "giornate");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,    "settimana", "settimane");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "quindicina");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,   "mese", "mesi");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter, "trimestre", "trimestri");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,    "anno", "anni");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,  "decennio", "decenni");
            b.Add(TermKind.Unit, (int)TimeUnit.Century, "secolo", "secoli");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend, "weekend", "fine settimana");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "notte", "notti");
            b.Add(TermKind.BusinessDay, "lavorativo", "lavorativi", "feriale", "feriali");
            b.Add(TermKind.Several, 3, "alcuni", "alcune", "diversi", "diverse", "qualche");
            b.Add(TermKind.Several, 2, "paio");
            b.Add(TermKind.HalfWord, "mezza", "mezzo");
            b.Add(TermKind.ToWord, "meno");   // "sette meno un quarto"
            b.Add(new TermInfo(TermKind.QuarterWord, 1, TermKind.Ordinal, 4), "quarto");

            b.Add(TermKind.Relative, (int)RelativeKind.This,     "questo", "questa", "questi", "queste");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,     "prossimo", "prossima", "prossimi", "próssimo", "seguente", "venturo");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,     "scorso", "scorsa", "scorsi", "scorse", "ultimo", "ultima", "ultimi", "ultime", "passato", "passata");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous, "precedente", "precedenti");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,  "corrente", "attuale", "stesso", "stessa");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "oggi");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow,           "domani");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "ieri");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "dopodomani");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "l'altro ieri", "avantieri");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "adesso", "in questo momento");

            b.Add(TermKind.Ago,     "fa");
            b.Add(new TermInfo(TermKind.Ago, 0, TermKind.Ordinal, 1), "prima");
            b.Add(TermKind.FromNow, "dopo", "da adesso");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later), "più tardi", "piu tardi");
            b.Add(new TermInfo(TermKind.Ago, 0, TermKind.Mod, (int)ModKind.Earlier), "prima nel", "più presto", "piu presto");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "mattina", "mattino", "mattinata");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "pomeriggio");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "sera", "serata");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "mezzogiorno");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "mezzanotte");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "stasera", "stanotte");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "pranzo");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "cena");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "colazione");

            b.Add(TermKind.AmPm, 0, "del mattino", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "del pomeriggio", "di sera", "pm", "p.m.");
            b.Add(TermKind.OClock, "in punto");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "fino a", "fino al");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ClockPrefix), "a", "ad", "al", "alle", "allo", "alla", "ai", "agli");   // "alle 10" as well as "dalle 5 alle 6"
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.AndWord), "e", "ed");
            b.Add(TermKind.RangeStart, 0, "da", "dalle", "a partire da", "a partire dal");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Since, TermKind.RangeStart, 0), "dal");
            b.Add(new TermInfo(TermKind.RangeStart, 1, TermKind.InPrefix, 0), "tra", "fra");   // "tra tre giorni" is in three days

            b.Add(TermKind.Filler, "di", "del", "della", "in", "nel", "nella", "per");
            b.Add(new TermInfo(TermKind.Article, 0, TermKind.Filler, 0), "il", "lo", "i", "gli", "l'", "un'", "dell'", "all'", "nell'");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "la", "le");   // "verso le tre"
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.Cardinal, 1), "un", "una");
            b.Add(new TermInfo(TermKind.Whole, 0, TermKind.Filler, 0), "tutto", "tutta", "tutti", "tutte", "intero", "intera");   // "the whole day" counts as one
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.InPrefix, 0), "in");
            b.Add(TermKind.InPrefix, 1, "entro");

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "prima di", "prima del", "entro");
            b.Add(TermKind.Mod, (int)ModKind.After, "dopo il", "dopo di", "più tardi di");
            b.Add(TermKind.Mod, (int)ModKind.Less,  "meno di");
            b.Add(TermKind.Mod, (int)ModKind.More,  "più di", "piu di");
            b.Add(TermKind.Mod, (int)ModKind.Start, "inizio", "inizio di", "principio di");
            b.Add(TermKind.Mod, (int)ModKind.End,   "fine", "fine di", "fine del");
            b.Add(TermKind.Mod, (int)ModKind.Mid,   "metà", "meta", "metà di");
            b.Add(TermKind.Mod, (int)ModKind.After,  "successivo a", "posteriore a", "dopo di il");
            b.Add(TermKind.Mod, (int)ModKind.Before, "precedente a", "anteriore a");
            b.Add(TermKind.Mod, (int)ModKind.Since,  "da allora", "non appena");
            b.Add(TermKind.Mod, (int)ModKind.Early,  "inizio di", "primi di");
            b.Add(TermKind.Mod, (int)ModKind.Late,   "fine di", "ultimi di");
            b.Add(TermKind.Mod, (int)ModKind.OrLater,   "o successivo", "e successivo", "o più tardi", "e più tardi",
                                                        "o dopo", "e dopo", "o piu tardi");
            b.Add(TermKind.Mod, (int)ModKind.OrEarlier, "o precedente", "e precedente", "o prima", "e prima");
            b.Add(TermKind.Approx, "circa", "intorno a", "verso", "approssimativamente");

            b.Add(TermKind.SetPrefix, 0, "ogni");
            b.Add(new TermInfo(TermKind.SetPrefix, 0, TermKind.Filler, 0), "tutti", "tutte");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "giornaliero", "quotidiano", "quotidianamente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "settimanale", "settimanalmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "mensile", "mensilmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "annuale", "annualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "trimestrale");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "primavera");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "estate");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "autunno");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "inverno");

                        b.Add(TermKind.Fiscal, 0, "solare");
            b.Add(TermKind.Fiscal, 1, "fiscale");
            b.Add(TermKind.Fiscal, 2, "scolastico", "accademico");
            b.Add(TermKind.QuarterMarker, 4, "t");
            b.Add(TermKind.QuarterMarker, 2, "s");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,      "capodanno", "primo dell'anno");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,   "san silvestro", "vigilia di capodanno");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,    "natale");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve, "vigilia di natale");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,       "pasqua");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday, "pasquetta", "lunedì dell'angelo");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,   "venerdì santo");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,    "ognissanti", "tutti i santi");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,   "san valentino");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,    "halloween");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,   "festa della mamma");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,   "festa del papà");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "festa del lavoro", "primo maggio");

            return b.Build(Language.Italian, dayMonthOrder: true, decimalComma: true, articleInDateSpan: false, articleInPeriodSpan: true, relativeAfterUnit: true, minutesFollowHour: true);
        }
    }
}
