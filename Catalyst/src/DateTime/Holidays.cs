using System;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>Computes the date of each known holiday for a given year, including the movable feasts.</summary>
    public static class Holidays
    {
        public static DateTime Resolve(HolidayKind kind, int year)
        {
            switch (kind)
            {
                case HolidayKind.NewYear:                 return new DateTime(year,  1,  1);
                case HolidayKind.Yuandan:                 return new DateTime(year,  1,  1);
                case HolidayKind.NewYearEve:              return new DateTime(year, 12, 31);
                case HolidayKind.Christmas:               return new DateTime(year, 12, 25);
                case HolidayKind.ChristmasEve:            return new DateTime(year, 12, 24);
                case HolidayKind.Boxing:                  return new DateTime(year, 12, 26);
                case HolidayKind.Valentines:              return new DateTime(year,  2, 14);
                case HolidayKind.AprilFools:              return new DateTime(year,  4,  1);
                case HolidayKind.Halloween:               return new DateTime(year, 10, 31);
                case HolidayKind.IndependenceDay:         return new DateTime(year,  7,  4);
                case HolidayKind.Groundhog:               return new DateTime(year,  2,  2);
                case HolidayKind.StPatricksDay:           return new DateTime(year,  3, 17);
                case HolidayKind.EarthDay:                return new DateTime(year,  4, 22);
                case HolidayKind.Juneteenth:              return new DateTime(year,  6, 19);
                case HolidayKind.FreedomDay:              return new DateTime(year,  6, 19);   // the other name for Juneteenth
                case HolidayKind.JubileeDay:              return new DateTime(year,  6, 19);
                case HolidayKind.InternationalWorkersDay: return new DateTime(year,  5,  1);
                case HolidayKind.VeteransDay:             return new DateTime(year, 11, 11);
                case HolidayKind.GermanUnityDay:          return new DateTime(year, 10,  3);
                case HolidayKind.BastilleDay:             return new DateTime(year,  7, 14);
                case HolidayKind.CanadaDay:               return new DateTime(year,  7,  1);
                case HolidayKind.AustraliaDay:            return new DateTime(year,  1, 26);
                case HolidayKind.AnzacDay:                return new DateTime(year,  4, 25);
                case HolidayKind.AllSaints:               return new DateTime(year, 11,  1);

                case HolidayKind.MartinLutherKingDay:     return NthWeekdayOfMonth(year,  1, DayOfWeek.Monday,   3);
                case HolidayKind.PresidentsDay:           return NthWeekdayOfMonth(year,  2, DayOfWeek.Monday,   3);
                case HolidayKind.MothersDay:              return NthWeekdayOfMonth(year,  5, DayOfWeek.Sunday,   2);
                case HolidayKind.MemorialDay:             return LastWeekdayOfMonth(year, 5, DayOfWeek.Monday);
                case HolidayKind.FathersDay:              return NthWeekdayOfMonth(year,  6, DayOfWeek.Sunday,   3);
                case HolidayKind.LaborDay:                return NthWeekdayOfMonth(year,  9, DayOfWeek.Monday,   1);
                case HolidayKind.ColumbusDay:             return NthWeekdayOfMonth(year, 10, DayOfWeek.Monday,   2);
                case HolidayKind.Thanksgiving:            return NthWeekdayOfMonth(year, 11, DayOfWeek.Thursday, 4);
                case HolidayKind.BlackFriday:             return NthWeekdayOfMonth(year, 11, DayOfWeek.Thursday, 4).AddDays(1);
                case HolidayKind.CyberMonday:             return NthWeekdayOfMonth(year, 11, DayOfWeek.Thursday, 4).AddDays(4);

                case HolidayKind.Easter:                  return Easter(year);
                case HolidayKind.EasterMonday:            return Easter(year).AddDays(1);
                case HolidayKind.GoodFriday:              return Easter(year).AddDays(-2);

                // Lunar feasts are approximated by their Gregorian date in recent years
                case HolidayKind.EidAlFitr:               return EidAlFitr(year);

                default:                                  return new DateTime(year, 1, 1);
            }
        }

        /// <summary>True when the holiday falls on the same calendar day every year.</summary>
        public static bool IsFixedDate(HolidayKind kind)
        {
            var a = Resolve(kind, 2020);
            var b = Resolve(kind, 2021);
            return a.Month == b.Month && a.Day == b.Day;
        }

        private static DateTime NthWeekdayOfMonth(int year, int month, DayOfWeek weekday, int n)
        {
            var first  = new DateTime(year, month, 1);
            int delta  = ((int)weekday - (int)first.DayOfWeek + 7) % 7;
            var result = first.AddDays(delta + (n - 1) * 7);

            if (result.Month != month) result = result.AddDays(-7);

            return result;
        }

        private static DateTime LastWeekdayOfMonth(int year, int month, DayOfWeek weekday)
        {
            var last  = new DateTime(year, month, DateTime.DaysInMonth(year, month));
            int delta = ((int)last.DayOfWeek - (int)weekday + 7) % 7;
            return last.AddDays(-delta);
        }

        /// <summary>The anonymous Gregorian algorithm.</summary>
        private static DateTime Easter(int year)
        {
            int a = year % 19;
            int b = year / 100;
            int c = year % 100;
            int d = b / 4;
            int e = b % 4;
            int f = (b + 8) / 25;
            int g = (b - f + 1) / 3;
            int h = (19 * a + b - d - g + 15) % 30;
            int i = c / 4;
            int k = c % 4;
            int l = (32 + 2 * e + 2 * i - h - k) % 7;
            int m = (a + 11 * h + 22 * l) / 451;
            int month = (h + l - 7 * m + 114) / 31;
            int day   = ((h + l - 7 * m + 114) % 31) + 1;

            return new DateTime(year, month, day);
        }

        private static DateTime EidAlFitr(int year)
        {
            // The Islamic year drifts about 11 days a year against the Gregorian one; anchored on 2020.
            var anchor = new DateTime(2020, 5, 24);

            // The mean lunar year runs a little long against the observed sightings, so the day is floored
            return anchor.AddDays((year - 2020) * 354.367 - 0.5).Date;
        }
    }
}
