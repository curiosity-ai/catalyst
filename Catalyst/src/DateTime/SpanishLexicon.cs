using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The Spanish vocabulary.</summary>
    public static class SpanishLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "enero", "ene");
            b.Add(TermKind.Month,  2, "febrero", "feb");
            b.Add(TermKind.Month,  3, "marzo", "mar");
            b.Add(TermKind.Month,  4, "abril", "abr");
            b.Add(TermKind.Month,  5, "mayo", "may");
            b.Add(TermKind.Month,  6, "junio", "jun");
            b.Add(TermKind.Month,  7, "julio", "jul");
            b.Add(TermKind.Month,  8, "agosto", "ago");
            b.Add(TermKind.Month,  9, "septiembre", "setiembre", "sep", "sept");
            b.Add(TermKind.Month, 10, "octubre", "oct");
            b.Add(TermKind.Month, 11, "noviembre", "nov");
            b.Add(TermKind.Month, 12, "diciembre", "dic");

            b.Add(TermKind.Weekday, 0, "domingo", "domingos", "dom");
            b.Add(TermKind.Weekday, 1, "lunes", "lun");
            b.Add(TermKind.Weekday, 2, "martes", "mar");
            b.Add(TermKind.Weekday, 3, "miércoles", "miercoles", "mié", "mie");
            b.Add(TermKind.Weekday, 4, "jueves", "jue");
            b.Add(TermKind.Weekday, 5, "viernes", "vie");
            b.Add(TermKind.Weekday, 6, "sábado", "sabado", "sábados", "sab", "sáb");

            b.Add(TermKind.Cardinal,  0, "cero");
            b.Add(TermKind.Cardinal,  1, "uno", "una", "un");
            b.Add(TermKind.Cardinal,  2, "dos");
            b.Add(TermKind.Cardinal,  3, "tres");
            b.Add(TermKind.Cardinal,  4, "cuatro");
            b.Add(TermKind.Cardinal,  5, "cinco");
            b.Add(TermKind.Cardinal,  6, "seis");
            b.Add(TermKind.Cardinal,  7, "siete");
            b.Add(TermKind.Cardinal,  8, "ocho");
            b.Add(TermKind.Cardinal,  9, "nueve");
            b.Add(TermKind.Cardinal, 10, "diez");
            b.Add(TermKind.Cardinal, 11, "once");
            b.Add(TermKind.Cardinal, 12, "doce");
            b.Add(TermKind.Cardinal, 13, "trece");
            b.Add(TermKind.Cardinal, 14, "catorce");
            b.Add(TermKind.Cardinal, 15, "quince");
            b.Add(TermKind.Cardinal, 16, "dieciséis", "dieciseis");
            b.Add(TermKind.Cardinal, 17, "diecisiete");
            b.Add(TermKind.Cardinal, 18, "dieciocho");
            b.Add(TermKind.Cardinal, 19, "diecinueve");
            b.Add(TermKind.Cardinal, 20, "veinte");
            b.Add(TermKind.Cardinal, 21, "veintiuno", "veintiún");
            b.Add(TermKind.Cardinal, 30, "treinta");
            b.Add(TermKind.Cardinal, 40, "cuarenta");
            b.Add(TermKind.Cardinal, 50, "cincuenta");
            b.Add(TermKind.Cardinal, 60, "sesenta");
            b.Add(TermKind.Cardinal, 70, "setenta");
            b.Add(TermKind.Cardinal, 80, "ochenta");
            b.Add(TermKind.Cardinal, 90, "noventa");
            b.Add(TermKind.Multiplier,     100, "cien", "ciento", "cientos");
            b.Add(TermKind.Multiplier,    1000, "mil");
            b.Add(TermKind.Multiplier, 1000000, "millón", "millon", "millones");

            b.Add(TermKind.Ordinal,  1, "primero", "primera", "primer");
            b.Add(TermKind.Ordinal,  2, "segundo", "segunda");
            b.Add(TermKind.Ordinal,  3, "tercero", "tercera", "tercer");
            b.Add(TermKind.Ordinal,  4, "cuarto", "cuarta");
            b.Add(TermKind.Ordinal,  5, "quinto", "quinta");
            b.Add(TermKind.Ordinal,  6, "sexto", "sexta");
            b.Add(TermKind.Ordinal,  7, "séptimo", "septimo");
            b.Add(TermKind.Ordinal,  8, "octavo", "octava");
            b.Add(TermKind.Ordinal,  9, "noveno", "novena");
            b.Add(TermKind.Ordinal, 10, "décimo", "decimo");
            b.Add(TermKind.OrdinalSuffix, "º", "ª", "er", "o", "a");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,    "segundo", "segundos", "seg");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,    "minuto", "minutos", "min");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "hora", "horas", "h", "hrs");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,       "día", "dia", "días", "dias", "d");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,      "semana", "semanas");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "quincena", "quincenas");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,     "mes", "meses");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter,   "trimestre", "trimestres");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,      "año", "ano", "años", "anos");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,    "década", "decada", "décadas");
            b.Add(TermKind.Unit, (int)TimeUnit.Century,   "siglo", "siglos");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend,   "fin de semana", "fines de semana");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "noche", "noches");
            b.Add(TermKind.BusinessDay, "laborable", "laborables", "hábil", "hábiles", "habiles");
            b.Add(TermKind.Several, 3, "unos", "unas", "varios", "varias", "algunos", "algunas");
            b.Add(TermKind.Several, 2, "par");
            b.Add(TermKind.HalfWord, "media", "medio", "y media");
            b.Add(TermKind.QuarterWord, "cuarto");

            b.Add(TermKind.Relative, (int)RelativeKind.This,      "este", "esta", "estos", "estas", "el presente");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,      "próximo", "proximo", "próxima", "proxima", "próximos", "siguiente", "siguientes", "que viene");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,      "pasado", "pasada", "pasados", "pasadas", "último", "ultimo", "última", "ultima");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous,  "anterior", "anteriores", "previo", "previa");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,   "actual", "corriente", "mismo", "misma");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "hoy");
            // "mañana" is both tomorrow and the morning; the grammar picks by context
            b.Add(new TermInfo(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow, TermKind.PartOfDay, (int)PartOfDayKind.Morning), "mañana", "manana");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "ayer");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "pasado mañana");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "anteayer", "antes de ayer");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "ahora", "ahora mismo", "en este momento");

            b.Add(TermKind.Ago,     "hace", "atrás", "atras", "antes");
            b.Add(TermKind.FromNow, "después", "despues", "más tarde", "mas tarde", "a partir de ahora");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "mañanas", "madrugada");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "tarde", "tardes");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "atardecer", "anochecer");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "mediodía", "mediodia");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "medianoche");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "esta noche");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "almuerzo", "comida");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "cena");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "desayuno");

            b.Add(TermKind.AmPm, 0, "de la mañana", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "de la tarde", "de la noche", "pm", "p.m.");
            b.Add(TermKind.OClock, "en punto");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "a", "al", "hasta", "hasta el");
            b.Add(TermKind.Connector, "y", "e");
            b.Add(TermKind.RangeStart, 0, "desde", "desde el", "a partir de", "a partir del", "comenzando");
            b.Add(TermKind.RangeStart, 1, "entre");

            b.Add(TermKind.Filler, "de", "del", "el", "la", "los", "las", "en", "por", "para", "un", "una");

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "antes de", "antes del", "no más tarde de");
            b.Add(TermKind.Mod, (int)ModKind.After,     "después de", "despues de", "más tarde que");
            b.Add(TermKind.Mod, (int)ModKind.Less,      "menos de");
            b.Add(TermKind.Mod, (int)ModKind.More,      "más de", "mas de");
            b.Add(TermKind.Mod, (int)ModKind.Start,     "principio", "principios", "inicio", "comienzo", "principios de");
            b.Add(TermKind.Mod, (int)ModKind.End,       "fin", "final", "finales", "fin de", "finales de");
            b.Add(TermKind.Mod, (int)ModKind.Mid,       "mediados", "mediados de", "medio de");
            b.Add(TermKind.Approx, "alrededor de", "aproximadamente", "cerca de", "sobre las", "unos");

            b.Add(TermKind.SetPrefix, 0, "cada", "todos", "todas");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "diario", "diaria", "diariamente", "a diario");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "semanal", "semanalmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "mensual", "mensualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "anual", "anualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "cada hora");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "trimestral", "trimestralmente");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "primavera");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "verano");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "otoño", "otono");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "invierno");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,      "año nuevo", "ano nuevo", "día de año nuevo");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,   "nochevieja", "víspera de año nuevo");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,    "navidad", "día de navidad");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve, "nochebuena", "víspera de navidad");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,       "pascua", "domingo de pascua", "domingo de resurrección");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday, "lunes de pascua");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,   "viernes santo");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,    "todos los santos", "día de todos los santos");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,   "san valentín", "día de san valentín");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,    "halloween");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,   "día de la madre");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,   "día del padre");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "día del trabajo", "día del trabajador");

            return b.Build(Language.Spanish, dayMonthOrder: true, decimalComma: true, articleInSpan: true);
        }
    }
}
