using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The Portuguese vocabulary.</summary>
    public static class PortugueseLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "janeiro", "jan");
            b.Add(TermKind.Month,  2, "fevereiro", "fev");
            b.Add(TermKind.Month,  3, "março", "marco", "mar");
            b.Add(TermKind.Month,  4, "abril", "abr");
            b.Add(TermKind.Month,  5, "maio", "mai");
            b.Add(TermKind.Month,  6, "junho", "jun");
            b.Add(TermKind.Month,  7, "julho", "jul");
            b.Add(TermKind.Month,  8, "agosto", "ago");
            b.Add(TermKind.Month,  9, "setembro", "set");
            b.Add(TermKind.Month, 10, "outubro", "out");
            b.Add(TermKind.Month, 11, "novembro", "nov");
            b.Add(TermKind.Month, 12, "dezembro");

            b.Add(TermKind.Weekday, 0, "domingo", "domingos", "dom");
            b.Add(TermKind.Weekday, 1, "segunda", "segunda-feira", "segundas", "segundas-feiras", "seg");
            b.Add(TermKind.Weekday, 2, "terça", "terca", "terça-feira", "terca-feira", "terças", "tercas", "terças-feiras", "ter");
            b.Add(TermKind.Weekday, 3, "quarta", "quarta-feira", "quartas", "quartas-feiras", "qua");
            b.Add(TermKind.Weekday, 4, "quinta", "quinta-feira", "quintas", "quintas-feiras", "qui");
            b.Add(TermKind.Weekday, 5, "sexta", "sexta-feira", "sextas", "sextas-feiras", "sex");
            b.Add(TermKind.Weekday, 6, "sábado", "sabado", "sábados", "sab", "sáb");

            b.Add(TermKind.Cardinal,  0, "zero");
            b.Add(TermKind.Cardinal,  1, "um", "uma");
            b.Add(TermKind.Cardinal,  2, "dois", "duas");
            b.Add(TermKind.Cardinal,  3, "três", "tres");
            b.Add(TermKind.Cardinal,  4, "quatro");
            b.Add(TermKind.Cardinal,  5, "cinco");
            b.Add(TermKind.Cardinal,  6, "seis");
            b.Add(TermKind.Cardinal,  7, "sete");
            b.Add(TermKind.Cardinal,  8, "oito");
            b.Add(TermKind.Cardinal,  9, "nove");
            b.Add(new TermInfo(TermKind.Cardinal, 10, TermKind.Month, 12), "dez");
            b.Add(TermKind.Cardinal, 11, "onze");
            b.Add(TermKind.Cardinal, 12, "doze");
            b.Add(TermKind.Cardinal, 13, "treze");
            b.Add(TermKind.Cardinal, 14, "catorze", "quatorze");
            b.Add(TermKind.Cardinal, 15, "quinze");
            b.Add(TermKind.Cardinal, 16, "dezesseis", "dezasseis");
            b.Add(TermKind.Cardinal, 17, "dezessete", "dezassete");
            b.Add(TermKind.Cardinal, 18, "dezoito");
            b.Add(TermKind.Cardinal, 19, "dezenove", "dezanove");
            b.Add(TermKind.Cardinal, 20, "vinte");
            b.Add(TermKind.Cardinal, 30, "trinta");
            b.Add(TermKind.Cardinal, 40, "quarenta");
            b.Add(TermKind.Cardinal, 50, "cinquenta");
            b.Add(TermKind.Cardinal, 60, "sessenta");
            b.Add(TermKind.Cardinal, 70, "setenta");
            b.Add(TermKind.Cardinal, 80, "oitenta");
            b.Add(TermKind.Cardinal, 90, "noventa");
            b.Add(TermKind.Multiplier,     100, "cem", "cento");
            b.Add(TermKind.Multiplier,    1000, "mil");
            b.Add(TermKind.Multiplier, 1000000, "milhão", "milhao", "milhões");

            b.Add(TermKind.Ordinal,  1, "primeiro", "primeira");
            
            b.Add(TermKind.Ordinal,  3, "terceiro", "terceira");
            
            b.Add(TermKind.Ordinal,  5, "quinto");
            b.Add(TermKind.Ordinal,  6, "sexto");
            b.Add(TermKind.Ordinal,  7, "sétimo", "setimo");
            b.Add(TermKind.Ordinal,  8, "oitavo");
            b.Add(TermKind.Ordinal,  9, "nono");
            b.Add(TermKind.Ordinal, 10, "décimo", "decimo");
            b.Add(TermKind.OrdinalSuffix, "º", "ª", "o", "a");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,  "segundos");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Second, TermKind.Ordinal, 2), "segundo");
            b.Add(new TermInfo(TermKind.Weekday, 1, TermKind.Unit, (int)TimeUnit.Second), "seg");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,  "minuto", "minutos", "min");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "hora", "horas", "h", "hrs");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,     "dia", "dias", "d");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,    "semana", "semanas");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "quinzena", "quinzenas");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,   "mês", "mes", "meses");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter, "trimestre", "trimestres");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,    "ano", "anos");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,  "década", "decada", "décadas");
            b.Add(TermKind.Unit, (int)TimeUnit.Century, "século", "seculo", "séculos");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend, "fim de semana", "fins de semana", "final de semana");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "noite", "noites");
            b.Add(TermKind.BusinessDay, "útil", "úteis", "uteis", "trabalho");
            b.Add(TermKind.Several, 3, "alguns", "algumas", "vários", "varias", "poucos");
            b.Add(TermKind.Several, 2, "par");
            b.Add(TermKind.HalfWord, "meia", "meio", "e meia");
            b.Add(new TermInfo(TermKind.QuarterWord, 1, TermKind.Ordinal, 4), "quarto");

            b.Add(TermKind.Relative, (int)RelativeKind.This,     "este", "esta", "estes", "estas", "esse", "essa");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,     "próximo", "proximo", "próxima", "proxima", "seguinte", "que vem");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,     "passado", "passada", "último", "ultimo", "última", "ultima");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous, "anterior", "anteriores", "prévio");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,  "atual", "corrente", "mesmo", "mesma");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "hoje");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow,           "amanhã", "amanha");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "ontem");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "depois de amanhã");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "anteontem", "antes de ontem");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "agora", "agora mesmo", "neste momento");

            b.Add(TermKind.Ago,     "atrás", "atras", "há", "ha");
            b.Add(TermKind.FromNow, "depois", "a partir de agora");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later), "mais tarde");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Earlier), "mais cedo");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "manhã", "manha", "manhãs", "madrugada");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "tarde", "tardes");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "meio-dia", "meio dia");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "meia-noite", "meia noite");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "esta noite", "hoje à noite");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "almoço", "almoco");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "jantar");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "café da manhã", "pequeno-almoço");

            b.Add(TermKind.AmPm, 0, "da manhã", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "da tarde", "da noite", "pm", "p.m.");
            b.Add(TermKind.OClock, "em ponto");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "a", "ao", "até", "ate", "até o");
            b.Add(TermKind.Connector, "e");
            b.Add(TermKind.RangeStart, 0, "desde", "a partir de", "começando");
            b.Add(TermKind.RangeStart, 1, "entre");

            b.Add(TermKind.Filler, "dos", "o", "os", "no", "na", "por", "para");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.Cardinal, 1), "um", "uma");
            b.Add(TermKind.Filler, "todo", "toda", "todos", "todas", "inteiro", "inteira");   // "the whole day" counts as one
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.RangeStart, 0), "de", "do");      // "2 de outubro" and "de 1/10 a 11/7"
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.InPrefix, 0), "em");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "da", "das", "as");
            b.Add(TermKind.InPrefix, 1, "dentro de", "dentro do");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ClockPrefix), "às", "à");   // "às 5" as well as "de 23 às 4"

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "antes de", "antes do", "no máximo até");
            b.Add(TermKind.Mod, (int)ModKind.After,  "depois de", "depois do", "após");
            b.Add(TermKind.Mod, (int)ModKind.Less,   "menos de");
            b.Add(TermKind.Mod, (int)ModKind.More,   "mais de");
            b.Add(TermKind.Mod, (int)ModKind.Start,  "início", "inicio", "começo", "princípio", "início de");
            b.Add(TermKind.Mod, (int)ModKind.End,    "fim", "final", "fim de", "final de");
            b.Add(TermKind.Mod, (int)ModKind.Mid,    "meados", "meados de", "meio de");
            b.Add(TermKind.Approx, "por volta de", "aproximadamente", "cerca de", "quase");

            b.Add(TermKind.SetPrefix, 0, "cada");
            b.Add(new TermInfo(TermKind.SetPrefix, 0, TermKind.Filler, 0), "todos", "todas", "todo", "toda");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "diário", "diaria", "diariamente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "semanal", "semanalmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "mensal", "mensalmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "anual", "anualmente");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "trimestral", "trimestralmente");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "primavera");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "verão", "verao");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "outono");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "inverno");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,      "ano novo", "dia de ano novo");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,   "véspera de ano novo", "réveillon");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,    "natal", "dia de natal");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve, "véspera de natal", "noite de natal");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,       "páscoa", "pascoa", "domingo de páscoa");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,   "sexta-feira santa");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,    "dia de todos os santos", "finados");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,   "dia dos namorados");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,    "halloween", "dia das bruxas");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,   "dia das mães");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,   "dia dos pais");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "dia do trabalho", "dia do trabalhador");

            return b.Build(Language.Portuguese, dayMonthOrder: true, decimalComma: true, articleInDateSpan: false, articleInPeriodSpan: false, relativeAfterUnit: true, minutesFollowHour: true);
        }
    }
}
