using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using P = Catalyst.PatternUnitPrototype;

namespace Catalyst.Presidio
{
    public static class PresidioAnalyzerExtensions
    {
        public static PresidioAnalyzer AddEmail(this PresidioAnalyzer analyzer)
        {
            var emailSpotter = new PatternSpotter(analyzer.Language, 0, "email", "EMAIL_ADDRESS");
            emailSpotter.NewPattern("Email", mp => mp.Add(new PatternUnit(P.Single().LikeEmail())));
            return analyzer.AddRecognizer(emailSpotter);
        }

        public static PresidioAnalyzer AddUrl(this PresidioAnalyzer analyzer)
        {
            var urlSpotter = new PatternSpotter(analyzer.Language, 0, "url", "URL");
            urlSpotter.NewPattern("URL", mp => mp.Add(new PatternUnit(P.Single().LikeURL())));
            return analyzer.AddRecognizer(urlSpotter);
        }

        public static PresidioAnalyzer AddPhone(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone", "PHONE_NUMBER");
            // Matches 555-123-4567 (Single token)
            phoneSpotter.NewPattern("Phone-US-Single", mp => mp.Add(new PatternUnit(P.Single().WithShape("999-999-9999"))));
            // Matches (555) 123-4567
            phoneSpotter.NewPattern("Phone-US-Parens", mp => mp.Add(
                new PatternUnit(P.Single().IsOpeningParenthesis()),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsClosingParenthesis()),
                new PatternUnit(P.Single().WithShape("999-9999"))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddCreditCard(this PresidioAnalyzer analyzer)
        {
            var ccSpotter = new PatternSpotter(analyzer.Language, 0, "cc", "CREDIT_CARD");
            // Matches 1234 5678 1234 5678
            ccSpotter.NewPattern("CC-4x4", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            return analyzer.AddRecognizer(ccSpotter);
        }

        public static PresidioAnalyzer AddIp(this PresidioAnalyzer analyzer)
        {
            var ipSpotter = new PatternSpotter(analyzer.Language, 0, "ip", "IP_ADDRESS");
            // Matches 192.168.1.1
            ipSpotter.NewPattern("IP-v4", mp => mp.Add(new PatternUnit(P.Single().WithShape("999.999.9.9"))));
            return analyzer.AddRecognizer(ipSpotter);
        }

        public static PresidioAnalyzer AddUsSsn(this PresidioAnalyzer analyzer)
        {
            var ssnSpotter = new PatternSpotter(analyzer.Language, 0, "us_ssn", "US_SSN");
            // Matches 123-45-6789
            ssnSpotter.NewPattern("US-SSN", mp => mp.Add(new PatternUnit(P.Single().WithShape("999-99-9999"))));
            return analyzer.AddRecognizer(ssnSpotter);
        }

        public static PresidioAnalyzer AddUsPassport(this PresidioAnalyzer analyzer)
        {
            var passportSpotter = new PatternSpotter(analyzer.Language, 0, "us_passport", "US_PASSPORT");
            // Matches 9 digits
            passportSpotter.NewPattern("US-Passport", mp => mp.Add(new PatternUnit(P.Single().IsNumeric().WithLength(9, 9))));
            return analyzer.AddRecognizer(passportSpotter);
        }

        public static PresidioAnalyzer AddIban(this PresidioAnalyzer analyzer)
        {
            var ibanSpotter = new PatternSpotter(analyzer.Language, 0, "iban", "IBAN_CODE");
            // Matches IBANs (e.g. GB12345678901234) as a single token.
            // Requires > 15 chars, starting with 2 letters.
            ibanSpotter.NewPattern("IBAN", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(15, 34).WithPrefix("GB,DE,FR,IT,ES")) // Example prefixes
            ));
            // Or simpler: just Length > 15 and is alphanumeric?
            // Catalyst doesn't have "StartsWithAlpha" easily without regex or prefixes.
            // Let's rely on length and char composition.
            return analyzer.AddRecognizer(ibanSpotter);
        }

        public static PresidioAnalyzer AddUsItin(this PresidioAnalyzer analyzer)
        {
            var itinSpotter = new PatternSpotter(analyzer.Language, 0, "us_itin", "US_ITIN");
            // Matches 9xx-7x-xxxx or 9xx-8x-xxxx
            // Shape is same as SSN: 999-99-9999
            // We can check if it starts with 9?
            // P.Single().WithShape("999-99-9999").WithPrefix("9")?
            // Catalyst Prefix check works on the token string.
            itinSpotter.NewPattern("US-ITIN", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("999-99-9999").WithPrefix("9"))
            ));
            return analyzer.AddRecognizer(itinSpotter);
        }

        public static PresidioAnalyzer AddCrypto(this PresidioAnalyzer analyzer)
        {
            var cryptoSpotter = new PatternSpotter(analyzer.Language, 0, "crypto", "CRYPTO");
            // Bitcoin addresses: 26-35 chars, start with 1, 3, or bc1
            cryptoSpotter.NewPattern("BTC-Legacy", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(26, 35).WithPrefix("1,3"))
            ));
            cryptoSpotter.NewPattern("BTC-Segwit", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(39, 62).WithPrefix("bc1"))
            ));
            return analyzer.AddRecognizer(cryptoSpotter);
        }

        public static PresidioAnalyzer AddUsDriverLicense(this PresidioAnalyzer analyzer)
        {
            var dlSpotter = new PatternSpotter(analyzer.Language, 0, "us_driver_license", "US_DRIVER_LICENSE");
            // Varies wildly. Common: 1 Letter + 7 Digits (NY, etc).
            // A1234567
            dlSpotter.NewPattern("DL-A1234567", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("X9999999"))
            ));
             dlSpotter.NewPattern("DL-A1234567-2", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("X99999999"))
            ));
            // 9 digits (Numeric)
            dlSpotter.NewPattern("DL-Digits", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(9, 9))
            ));
            return analyzer.AddRecognizer(dlSpotter);
        }

        public static PresidioAnalyzer AddAllRecognizers(this PresidioAnalyzer analyzer)
        {
            return analyzer
                .AddEmail()
                .AddUrl()
                .AddPhone()
                .AddCreditCard()
                .AddIp()
                .AddUsSsn()
                .AddUsPassport()
                .AddIban()
                .AddUsItin()
                .AddCrypto()
                .AddUsDriverLicense();
        }
    }
}
