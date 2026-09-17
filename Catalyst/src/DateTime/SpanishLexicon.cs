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
            b.Add(TermKind.Cardinal, 20, "veinte", "veinti");   // "veinticuatro" elides the tens
            b.Add(TermKind.Cardinal, 21, "veintiuno", "veintiún", "veintiuna");
            b.Add(TermKind.Cardinal, 22, "veintidós", "veintidos");
            b.Add(TermKind.Cardinal, 23, "veintitrés", "veintitres");
            b.Add(TermKind.Cardinal, 24, "veinticuatro");
            b.Add(TermKind.Cardinal, 25, "veinticinco");
            b.Add(TermKind.Cardinal, 26, "veintiséis", "veintiseis");
            b.Add(TermKind.Cardinal, 27, "veintisiete");
            b.Add(TermKind.Cardinal, 28, "veintiocho");
            b.Add(TermKind.Cardinal, 29, "veintinueve");
            b.Add(TermKind.Cardinal, 30, "treinta");
            b.Add(TermKind.Cardinal, 40, "cuarenta");
            b.Add(TermKind.Cardinal, 50, "cincuenta");
            b.Add(TermKind.Cardinal, 60, "sesenta");
            b.Add(TermKind.Cardinal, 70, "setenta");
            b.Add(TermKind.Cardinal, 80, "ochenta");
            b.Add(TermKind.Cardinal, 90, "noventa");
            b.Add(TermKind.Multiplier,     100, "cien", "ciento", "cientos");
            b.Add(TermKind.Cardinal,  200, "doscientos");
            b.Add(TermKind.Cardinal,  300, "trescientos");
            b.Add(TermKind.Cardinal,  400, "cuatrocientos");
            b.Add(TermKind.Cardinal,  500, "quinientos");
            b.Add(TermKind.Cardinal,  600, "seiscientos");
            b.Add(TermKind.Cardinal,  700, "setecientos");
            b.Add(TermKind.Cardinal,  800, "ochocientos");
            b.Add(TermKind.Cardinal,  900, "novecientos");
            b.Add(TermKind.Multiplier,    1000, "mil");
            b.Add(TermKind.Multiplier, 1000000, "millón", "millon", "millones");

            b.Add(TermKind.Ordinal,  1, "primero", "primera", "primer", "primeros", "primeras");
            b.Add(TermKind.Ordinal,  2, "segunda", "segundos", "segundas");
            b.Add(TermKind.Ordinal,  3, "tercero", "tercera", "tercer", "terceros", "terceras");
            b.Add(TermKind.Ordinal,  4, "cuarta");
            b.Add(TermKind.Ordinal,  5, "quinto", "quinta");
            b.Add(TermKind.Ordinal,  6, "sexto", "sexta");
            b.Add(TermKind.Ordinal,  7, "séptimo", "septimo");
            b.Add(TermKind.Ordinal,  8, "octavo", "octava");
            b.Add(TermKind.Ordinal,  9, "noveno", "novena");
            b.Add(TermKind.Ordinal, 10, "décimo", "decimo");
            b.Add(TermKind.OrdinalSuffix, "º", "ª", "er", "o", "a", "ro", "do", "to", "mo", "vo", "no");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,    "segundos", "seg");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Second, TermKind.Ordinal, 2), "segundo");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,    "minuto", "minutos", "min", "mins");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "hora", "horas", "h", "hrs", "hra", "hras");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,       "día", "dia", "días", "dias", "d");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,      "semana", "semanas");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "quincena", "quincenas");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,     "mes", "meses");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter,   "trimestre", "trimestres");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,      "año", "ano", "años", "anos");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,    "década", "decada", "décadas", "decenio", "decenios");
            b.Add(TermKind.Unit, (int)TimeUnit.Century,   "siglo", "siglos");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend,   "fin de semana", "fines de semana", "finde", "findes");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "noche", "noches");
            b.Add(TermKind.BusinessDay, "laborable", "laborables", "hábil", "hábiles", "habiles");
            b.Add(TermKind.Several, 3, "unas", "varios", "varias", "algunos", "algunas");
            b.Add(TermKind.Several, 2, "par");
            b.Add(TermKind.Several, 1, "otro", "otra");
            b.Add(TermKind.HalfWord, "media", "medio");
            b.Add(TermKind.ToWord, "menos");   // "siete menos cuarto"
            b.Add(new TermInfo(TermKind.QuarterWord, 1, TermKind.Ordinal, 4), "cuarto");

            b.Add(TermKind.Relative, (int)RelativeKind.This,      "este", "esta", "estos", "estas", "el presente");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,      "próximo", "proximo", "próxima", "proxima", "próximos", "proximos", "próximas", "proximas",
                                                                 "siguiente", "siguientes", "que viene", "entrante", "entrantes");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,      "pasado", "pasada", "pasados", "pasadas", "último", "ultimo", "última", "ultima", "últimos", "ultimos", "últimas", "ultimas");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous,  "anterior", "anteriores", "previo", "previa");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,   "ese", "esa", "esos", "esas",   "actual", "corriente", "mismo", "misma");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "hoy");
            // "mañana" is both tomorrow and the morning; the grammar picks by context
            b.Add(new TermInfo(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow, TermKind.PartOfDay, (int)PartOfDayKind.Morning), "mañana", "manana");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "ayer");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "pasado mañana");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "anteayer", "antes de ayer");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "ahora", "ahora mismo", "en este momento");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.EndOfDay,           "fin de día", "fin del día", "final de día", "final del día",
                                                                               "fin de dia", "fin del dia", "final de dia", "final del dia");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.NextDay,            "el día siguiente", "el dia siguiente", "día siguiente", "dia siguiente");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.PriorDay,           "el día anterior", "el dia anterior", "día anterior", "dia anterior");

            b.Add(TermKind.Ago,     "hace", "atrás", "atras", "antes");
            b.Add(TermKind.FromNow, "después", "despues", "a partir de ahora");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later),   "más tarde", "mas tarde");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Earlier), "más temprano", "mas temprano", "más pronto");
            b.Add(TermKind.Mod, (int)ModKind.Early, "temprano");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "mañanas", "madrugada");
            // "la tarde" runs from four to eight in the suite, which is the evening slot
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "tarde", "tardes");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "atardecer", "anochecer");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "mediodía", "mediodia");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "medianoche");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "esta noche");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.LastNight, "anoche", "ayer por la noche", "ayer noche");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "almuerzo", "comida");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "cena");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "desayuno");

            b.Add(TermKind.AmPm, 0, "de la mañana", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "de la tarde", "de la noche", "pm", "p.m.");
            b.Add(TermKind.OClock, "en punto");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "a", "al");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.Mod, (int)ModKind.Before), "hasta", "hasta el", "hasta las", "hasta la");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.AndWord), "y", "e");
            b.Add(TermKind.RangeStart, 0, "a partir de", "a partir del", "a partir de las");
            // These open a range, and on their own say everything after the day they name
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.After, TermKind.RangeStart, 0),
                  "a primeros de", "comenzando", "empezando", "empienzando", "comienzo de");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.RangeStart, 0), "de", "del");   // "de 1/10 a 11/7"
            b.Add(TermKind.RangeStart, 1, "entre");

            b.Add(TermKind.Filler, "en", "por", "para");
            b.Add(new TermInfo(TermKind.Article, 0, TermKind.Filler, 0), "el", "los");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "la", "las");   // "cerca de las tres"
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.Cardinal, 1), "un", "una");
            b.Add(new TermInfo(TermKind.Whole, 0, TermKind.Filler, 0), "todo", "toda", "todos", "todas", "entero", "entera");   // "the whole day" counts as one
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.InPrefix, 0), "en");
            b.Add(TermKind.InPrefix, 1, "dentro de", "dentro del");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "de las", "de la");   // glue that can introduce a clock
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ClockPrefix), "a las", "a la");   // "de las 5 a las 6"

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "antes de", "antes del", "no más tarde de", "no más tarde que", "no mas tarde que");
            b.Add(TermKind.Mod, (int)ModKind.After,     "después de", "despues de", "después del", "despues del", "después de las", "más tarde que", "posterior a", "posterior de",
                                                        "posterior al");
            b.Add(TermKind.Mod, (int)ModKind.Before,    "anterior a", "anterior de", "anterior al", "más temprano que", "previo a");
            b.Add(TermKind.Mod, (int)ModKind.Less,      "menos de");
            b.Add(TermKind.Mod, (int)ModKind.More,      "más de", "mas de");
            b.Add(TermKind.Mod, (int)ModKind.Start,     "principio", "principios", "inicio", "inicios", "comienzo", "comienzos",
                                                        "inicia", "comienza", "empieza",
                                                        "principios de", "inicios de", "comienzos de");
            b.Add(TermKind.Mod, (int)ModKind.End,       "fin", "final", "finales", "fin de", "finales de");
            b.Add(TermKind.Mod, (int)ModKind.Mid,       "mediados", "mediados de", "medio de", "medianos", "medianos de");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Since, TermKind.RangeStart, 0), "desde", "desde el");
            b.Add(TermKind.Mod, (int)ModKind.Since,     "desde entonces", "tan pronto como", "a partir del momento",
                                                        "tan temprano como", "cualquier tiempo a partir de", "cualquier momento a partir de");
            b.Add(TermKind.Mod, (int)ModKind.Until,     "tan tarde como");
            b.Add(TermKind.Mod, (int)ModKind.OrLater,   "o posterior", "y posterior", "o más tarde", "y más tarde",
                                                        "o mas tarde", "y mas tarde", "o después", "y después", "o despues", "y despues");
            b.Add(TermKind.Mod, (int)ModKind.OrEarlier, "o anterior", "y anterior", "o antes", "y antes",
                                                        "o más temprano", "y más temprano");
            b.Add(TermKind.Approx, "alrededor de", "aproximadamente", "cerca de", "sobre las");
            b.Add(new TermInfo(TermKind.Approx, 0, TermKind.Several, 3), "unos");

            b.Add(TermKind.SetPrefix, 0, "cada");
            b.Add(new TermInfo(TermKind.SetPrefix, 0, TermKind.Filler, 0), "todos", "todas");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "diario", "diaria", "diariamente", "a diario");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "semanal", "semanalmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "mensual", "mensualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "anual", "anualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "cada hora");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "trimestral", "trimestralmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.HalfYear, "semestral", "semestrales", "semestralmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Fortnight, "quincenal", "quincenales", "quincenalmente", "bimensual", "bimensuales");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "primavera");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "verano");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "otoño", "otono");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "invierno");

                        b.Add(TermKind.Fiscal, 0, "calendario");
            b.Add(TermKind.Fiscal, 1, "fiscal");
            b.Add(TermKind.Fiscal, 2, "escolar", "lectivo", "académico", "academico");
            b.Add(TermKind.QuarterMarker, 4, "t");
            b.Add(TermKind.QuarterMarker, 2, "s");

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
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "día del trabajo", "día del trabajador",
                                                                              "día internacional de los trabajadores", "día internacional del trabajo");
            b.Add(TermKind.Holiday, (int)HolidayKind.IndependenceDay, "día de independencia", "día de la independencia");
            b.Add(TermKind.Holiday, (int)HolidayKind.EarthDay,     "día de la tierra");
            b.Add(TermKind.Holiday, (int)HolidayKind.StPatricksDay, "día de san patricio", "san patricio");
            b.Add(TermKind.Holiday, (int)HolidayKind.Juneteenth,   "juneteenth");
            b.Add(TermKind.Holiday, (int)HolidayKind.BlackFriday,  "viernes negro", "black friday");
            b.Add(TermKind.Holiday, (int)HolidayKind.Thanksgiving, "día de acción de gracias", "acción de gracias");
            b.Add(TermKind.Holiday, (int)HolidayKind.Epiphany,     "día de reyes", "reyes magos", "epifanía");
            b.Add(TermKind.Holiday, (int)HolidayKind.PalmSunday,   "domingo de ramos");
            b.Add(TermKind.Holiday, (int)HolidayKind.Pentecost,    "pentecostés", "pentecostes");

            return b.Build(Language.Spanish, dayMonthOrder: true, decimalComma: true, articleInDateSpan: false, articleInPeriodSpan: false, relativeAfterUnit: true, minutesFollowHour: true);
        }
    }
}
