using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>The French vocabulary.</summary>
    public static class FrenchLexicon
    {
        private static readonly Lazy<Lexicon> _lexicon = new Lazy<Lexicon>(Build, true);

        public static Lexicon Get() => _lexicon.Value;

        private static Lexicon Build()
        {
            var b = new LexiconBuilder();

            b.Add(TermKind.Month,  1, "janvier", "jan", "janv");
            b.Add(TermKind.Month,  2, "février", "fevrier", "fév", "fev", "févr");
            b.Add(TermKind.Month,  3, "mars", "mar");
            b.Add(TermKind.Month,  4, "avril", "avr");
            b.Add(TermKind.Month,  5, "mai");
            b.Add(TermKind.Month,  6, "juin", "jun");
            b.Add(TermKind.Month,  7, "juillet", "juil");
            b.Add(TermKind.Month,  8, "août", "aout", "aoû");
            b.Add(TermKind.Month,  9, "septembre", "sep");
            b.Add(TermKind.Month, 10, "octobre", "oct");
            b.Add(TermKind.Month, 11, "novembre", "nov");
            b.Add(TermKind.Month, 12, "décembre", "decembre", "déc", "dec");

            b.Add(TermKind.Weekday, 0, "dimanche", "dimanches", "dim");
            b.Add(TermKind.Weekday, 1, "lundi", "lundis", "lun");
            b.Add(TermKind.Weekday, 2, "mardi", "mardis", "mar");
            b.Add(TermKind.Weekday, 3, "mercredi", "mercredis", "mer");
            b.Add(TermKind.Weekday, 4, "jeudi", "jeudis", "jeu");
            b.Add(TermKind.Weekday, 5, "vendredi", "vendredis", "ven");
            b.Add(TermKind.Weekday, 6, "samedi", "samedis", "sam");

            b.Add(TermKind.Cardinal,  0, "zéro", "zero");
            b.Add(TermKind.Cardinal,  1, "un", "une");
            b.Add(TermKind.Cardinal,  2, "deux");
            b.Add(TermKind.Cardinal,  3, "trois");
            b.Add(TermKind.Cardinal,  4, "quatre");
            b.Add(TermKind.Cardinal,  5, "cinq");
            b.Add(TermKind.Cardinal,  6, "six");
            b.Add(new TermInfo(TermKind.Cardinal, 7, TermKind.Month, 9), "sept");
            b.Add(TermKind.Cardinal,  8, "huit");
            b.Add(TermKind.Cardinal,  9, "neuf");
            b.Add(TermKind.Cardinal, 10, "dix");
            b.Add(TermKind.Cardinal, 11, "onze");
            b.Add(TermKind.Cardinal, 12, "douze");
            b.Add(TermKind.Cardinal, 13, "treize");
            b.Add(TermKind.Cardinal, 14, "quatorze");
            b.Add(TermKind.Cardinal, 15, "quinze");
            b.Add(TermKind.Cardinal, 16, "seize");
            b.Add(TermKind.Cardinal, 17, "dix-sept");
            b.Add(TermKind.Cardinal, 18, "dix-huit");
            b.Add(TermKind.Cardinal, 19, "dix-neuf");
            b.Add(TermKind.Cardinal, 20, "vingt");
            b.Add(TermKind.Cardinal, 30, "trente");
            b.Add(TermKind.Cardinal, 40, "quarante");
            b.Add(TermKind.Cardinal, 50, "cinquante");
            b.Add(TermKind.Cardinal, 60, "soixante");
            b.Add(TermKind.Cardinal, 80, "quatre-vingt", "quatre-vingts");
            b.Add(TermKind.Multiplier,     100, "cent", "cents");
            b.Add(TermKind.Multiplier,    1000, "mille");
            b.Add(TermKind.Multiplier, 1000000, "million", "millions");

            b.Add(TermKind.Ordinal,  1, "premier", "première", "premiere", "1er");
            b.Add(TermKind.Ordinal,  2, "deuxième", "deuxieme", "second");
            b.Add(TermKind.Ordinal,  3, "troisième", "troisieme");
            b.Add(TermKind.Ordinal,  4, "quatrième", "quatrieme");
            b.Add(TermKind.Ordinal,  5, "cinquième", "cinquieme");
            b.Add(TermKind.Ordinal,  6, "sixième", "sixieme");
            b.Add(TermKind.Ordinal,  7, "septième", "septieme");
            b.Add(TermKind.Ordinal,  8, "huitième", "huitieme");
            b.Add(TermKind.Ordinal,  9, "neuvième", "neuvieme");
            b.Add(TermKind.Ordinal, 10, "dixième", "dixieme");
            b.Add(TermKind.OrdinalSuffix, "er", "ère", "ere", "ème", "eme", "e");

            b.Add(TermKind.Unit, (int)TimeUnit.Second,    "secondes", "sec");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Second, TermKind.Ordinal, 2), "seconde");
            b.Add(TermKind.Unit, (int)TimeUnit.Minute,    "minute", "minutes", "min");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Hour, TermKind.OClock), "heure", "heures", "h", "hr");
            b.Add(TermKind.Unit, (int)TimeUnit.Day,       "jour", "jours", "journée", "journee", "journées");
            b.Add(TermKind.Unit, (int)TimeUnit.Week,      "semaine", "semaines");
            b.Add(TermKind.Unit, (int)TimeUnit.Fortnight, "quinzaine");
            b.Add(TermKind.Unit, (int)TimeUnit.Month,     "mois");
            b.Add(TermKind.Unit, (int)TimeUnit.Quarter,   "trimestre", "trimestres");
            b.Add(TermKind.Unit, (int)TimeUnit.Year,      "an", "ans", "année", "annee", "années", "annees");
            b.Add(TermKind.Unit, (int)TimeUnit.Decade,    "décennie", "decennie", "décennies");
            b.Add(TermKind.Unit, (int)TimeUnit.Century,   "siècle", "siecle", "siècles");
            b.Add(TermKind.Unit, (int)TimeUnit.Weekend,   "week-end", "weekend", "week-ends", "weekends", "fin de semaine");
            b.Add(new TermInfo(TermKind.Unit, (int)TimeUnit.Night, TermKind.PartOfDay, (int)PartOfDayKind.Night), "nuit", "nuits");
            b.Add(TermKind.BusinessDay, "ouvrable", "ouvrables", "ouvré", "ouvrés");
            b.Add(TermKind.Several, 3, "quelques", "plusieurs", "certains");
            b.Add(TermKind.Several, 2, "couple");
            b.Add(TermKind.HalfWord, "demi", "demie");
            b.Add(TermKind.ToWord, "moins");   // "sept heures moins le quart"
            b.Add(TermKind.QuarterWord, "quart");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow, "lendemain");

            b.Add(TermKind.Relative, (int)RelativeKind.This,      "ce", "cet", "cette", "ces");
            b.Add(TermKind.Relative, (int)RelativeKind.Next,      "prochain", "prochaine", "prochains", "prochaines", "suivant", "suivante");
            b.Add(TermKind.Relative, (int)RelativeKind.Last,      "dernier", "dernière", "derniere", "derniers", "dernières", "passé", "passée", "passés");
            b.Add(TermKind.Relative, (int)RelativeKind.Previous,  "précédent", "precedent", "précédente", "avant-dernier");
            b.Add(TermKind.Relative, (int)RelativeKind.Current,   "courant", "courante", "actuel", "actuelle", "même", "meme");

            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Today,              "aujourd'hui", "aujourdhui");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Tomorrow,           "demain");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Yesterday,          "hier");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayAfterTomorrow,   "après-demain", "apres-demain");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.DayBeforeYesterday, "avant-hier");
            b.Add(TermKind.SpecialDay, (int)SpecialDayKind.Now,                "maintenant", "à l'instant", "en ce moment");

            b.Add(TermKind.Ago,     "il y a", "auparavant");
            b.Add(new TermInfo(TermKind.Ago, 0, TermKind.Mod, (int)ModKind.Earlier), "plus tôt", "plus tot");
            b.Add(TermKind.FromNow, "après", "apres", "dans le futur");
            b.Add(new TermInfo(TermKind.FromNow, 0, TermKind.Mod, (int)ModKind.Later), "plus tard");

            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Morning,   "matin", "matinée", "matinee", "matins");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Afternoon, "après-midi", "apres-midi", "aprèsmidi");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Evening,   "soir", "soirée", "soiree", "soirs");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Noon,      "midi");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Midnight,  "minuit");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Tonight,   "ce soir", "cette nuit");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Lunch,     "déjeuner", "dejeuner");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Dinner,    "dîner", "diner", "souper");
            b.Add(TermKind.PartOfDay, (int)PartOfDayKind.Breakfast, "petit déjeuner", "petit-déjeuner");

            b.Add(TermKind.AmPm, 0, "du matin", "am", "a.m.");
            b.Add(TermKind.AmPm, 1, "de l'après-midi", "du soir", "pm", "p.m.");
            b.Add(TermKind.OClock, "heures pile");

            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ToWord), "jusqu'à", "jusqu'au", "jusque");
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.ClockPrefix), "à", "a", "au");   // "à 10h" as well as "de 5 à 6"
            b.Add(new TermInfo(TermKind.Connector, 0, TermKind.AndWord), "et");
            b.Add(TermKind.RangeStart, 0, "dès", "des", "à partir de", "à partir du", "commençant");
            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Since, TermKind.RangeStart, 0), "depuis");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.RangeStart, 0), "de", "du");
            b.Add(TermKind.RangeStart, 1, "entre");

            b.Add(TermKind.Filler, "d'", "dans", "en", "sur", "pour", "par");
            b.Add(new TermInfo(TermKind.Article, 0, TermKind.Filler, 0), "le", "l'");
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.ClockPrefix), "la", "les");   // "vers les trois heures"
            b.Add(new TermInfo(TermKind.Whole, 0, TermKind.Filler, 0), "tout", "toute", "toutes", "entier", "entière");   // "the whole day" counts as one
            b.Add(new TermInfo(TermKind.Filler, 0, TermKind.InPrefix, 0), "dans", "en");
            b.Add(TermKind.InPrefix, 0, "d'ici");
            b.Add(TermKind.InPrefix, 1, "en moins de");

            b.Add(new TermInfo(TermKind.Mod, (int)ModKind.Before, TermKind.ToWord), "avant", "avant le", "au plus tard");
            b.Add(TermKind.Mod, (int)ModKind.After,     "après le", "apres le", "plus tard que");
            b.Add(TermKind.Mod, (int)ModKind.Less,      "moins de");
            b.Add(TermKind.Mod, (int)ModKind.More,      "plus de");
            b.Add(TermKind.Mod, (int)ModKind.Start,     "début", "debut", "début de", "début du");
            b.Add(TermKind.Mod, (int)ModKind.End,       "fin", "fin de", "fin du");
            b.Add(TermKind.Mod, (int)ModKind.Mid,       "mi", "milieu", "milieu de", "milieu du", "mi-");
            b.Add(TermKind.Mod, (int)ModKind.After,     "postérieur à", "posterieur a", "après");
            b.Add(TermKind.Mod, (int)ModKind.Before,    "antérieur à", "anterieur a", "plus tôt que");
            b.Add(TermKind.Mod, (int)ModKind.Since,     "depuis lors", "dès que");
            b.Add(TermKind.Mod, (int)ModKind.Early,     "tôt dans", "tot dans");
            b.Add(TermKind.Mod, (int)ModKind.OrLater,   "ou plus tard", "et plus tard", "ou après", "et après",
                                                        "ou ultérieur", "ou apres", "et apres");
            b.Add(TermKind.Mod, (int)ModKind.OrEarlier, "ou plus tôt", "et plus tôt", "ou avant", "et avant",
                                                        "ou antérieur", "ou plus tot");
            b.Add(TermKind.Approx, "environ", "vers", "aux alentours de", "à peu près");

            b.Add(TermKind.SetPrefix, 0, "chaque", "tous");
            b.Add(new TermInfo(TermKind.SetPrefix, 0, TermKind.Filler, 0), "toutes", "tout");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Day,     "quotidien", "quotidienne", "quotidiennement");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Week,    "hebdomadaire", "hebdomadairement");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Month,   "mensuel", "mensuelle", "mensuellement");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Year,    "annuel", "annuelle", "annuellement");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Hour,    "horaire");
            b.Add(TermKind.SetFrequency, (int)TimeUnit.Quarter, "trimestriel", "trimestrielle");

            b.Add(TermKind.Season, (int)SeasonKind.Spring, "printemps");
            b.Add(TermKind.Season, (int)SeasonKind.Summer, "été", "ete");
            b.Add(TermKind.Season, (int)SeasonKind.Fall,   "automne");
            b.Add(TermKind.Season, (int)SeasonKind.Winter, "hiver");

                        b.Add(TermKind.Fiscal, 0, "civile", "calendaire");
            b.Add(TermKind.Fiscal, 1, "fiscale", "fiscal");
            b.Add(TermKind.Fiscal, 2, "scolaire", "universitaire");
            b.Add(TermKind.QuarterMarker, 4, "t");
            b.Add(TermKind.QuarterMarker, 2, "s");

            b.Add(TermKind.Holiday, (int)HolidayKind.NewYear,      "nouvel an", "jour de l'an", "premier de l'an");
            b.Add(TermKind.Holiday, (int)HolidayKind.NewYearEve,   "réveillon du nouvel an", "saint-sylvestre");
            b.Add(TermKind.Holiday, (int)HolidayKind.Christmas,    "noël", "noel", "jour de noël");
            b.Add(TermKind.Holiday, (int)HolidayKind.ChristmasEve, "réveillon de noël", "veille de noël");
            b.Add(TermKind.Holiday, (int)HolidayKind.Easter,       "pâques", "paques");
            b.Add(TermKind.Holiday, (int)HolidayKind.EasterMonday, "lundi de pâques");
            b.Add(TermKind.Holiday, (int)HolidayKind.GoodFriday,   "vendredi saint");
            b.Add(TermKind.Holiday, (int)HolidayKind.BastilleDay,  "fête nationale", "quatorze juillet");
            b.Add(TermKind.Holiday, (int)HolidayKind.AllSaints,    "toussaint");
            b.Add(TermKind.Holiday, (int)HolidayKind.Valentines,   "saint-valentin", "la saint-valentin");
            b.Add(TermKind.Holiday, (int)HolidayKind.Halloween,    "halloween");
            b.Add(TermKind.Holiday, (int)HolidayKind.MothersDay,   "fête des mères");
            b.Add(TermKind.Holiday, (int)HolidayKind.FathersDay,   "fête des pères");
            b.Add(TermKind.Holiday, (int)HolidayKind.InternationalWorkersDay, "fête du travail");

            return b.Build(Language.French, dayMonthOrder: true, decimalComma: true, articleInDateSpan: false, articleInPeriodSpan: false, relativeAfterUnit: true, minutesFollowHour: true);
        }
    }
}
