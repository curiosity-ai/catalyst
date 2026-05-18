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
            // Matches 192.168.1.1 or 10.0.0.1
            ipSpotter.NewPattern("IP-LikeURL", mp => mp.Add(new PatternUnit(P.Single().LikeURL())));
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

            // WithPrefix only accepts one string. We need multiple patterns.
            // Or rely on generic length + letter check.
            // Let's add patterns for each common prefix group or generic.

            foreach(var prefix in new[] { "GB", "DE", "FR", "IT", "ES" })
            {
                ibanSpotter.NewPattern($"IBAN-{prefix}", mp => mp.Add(
                    new PatternUnit(P.Single().IsLetterOrDigit().WithLength(15, 34).WithPrefix(prefix))
                ));
            }
            return analyzer.AddRecognizer(ibanSpotter);
        }

        public static PresidioAnalyzer AddUsItin(this PresidioAnalyzer analyzer)
        {
            var itinSpotter = new PatternSpotter(analyzer.Language, 0, "us_itin", "US_ITIN");
            itinSpotter.NewPattern("US-ITIN", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("999-99-9999").WithPrefix("9"))
            ));
            return analyzer.AddRecognizer(itinSpotter);
        }

        public static PresidioAnalyzer AddCrypto(this PresidioAnalyzer analyzer)
        {
            var cryptoSpotter = new PatternSpotter(analyzer.Language, 0, "crypto", "CRYPTO");

            // BTC Legacy: 1 or 3
            cryptoSpotter.NewPattern("BTC-Legacy-1", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(26, 35).WithPrefix("1"))
            ));
            cryptoSpotter.NewPattern("BTC-Legacy-3", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(26, 35).WithPrefix("3"))
            ));

            cryptoSpotter.NewPattern("BTC-Segwit", mp => mp.Add(
                new PatternUnit(P.Single().IsLetterOrDigit().WithLength(39, 62).WithPrefix("bc1"))
            ));
            return analyzer.AddRecognizer(cryptoSpotter);
        }

        public static PresidioAnalyzer AddUsDriverLicense(this PresidioAnalyzer analyzer)
        {
            var dlSpotter = new PatternSpotter(analyzer.Language, 0, "us_driver_license", "US_DRIVER_LICENSE");
            dlSpotter.NewPattern("DL-A1234567", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("X9999999"))
            ));
             dlSpotter.NewPattern("DL-A1234567-2", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("X99999999"))
            ));
            dlSpotter.NewPattern("DL-Digits", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(9, 9))
            ));
            return analyzer.AddRecognizer(dlSpotter);
        }

        public static PresidioAnalyzer AddUsBankNumber(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "us_bank", "US_BANK_NUMBER");
            // 9 digits
            spotter.NewPattern("US-Bank", mp => mp.Add(new PatternUnit(P.Single().IsNumeric().WithLength(9, 9))));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddUkNhs(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "uk_nhs", "UK_NHS");
            // 10 digits
            spotter.NewPattern("UK-NHS-Single", mp => mp.Add(new PatternUnit(P.Single().IsNumeric().WithLength(10, 10))));
            // Formatted: 123 456 7890 (3, 3, 4)
            spotter.NewPattern("UK-NHS-Formatted", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddEsNif(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "es_nif", "ES_NIF");
            // 12345678Z -> 99999999X
            spotter.NewPattern("ES-NIF-Standard", mp => mp.Add(new PatternUnit(P.Single().WithShape("99999999X"))));
            // X1234567Z -> X9999999X (NIE)
            spotter.NewPattern("ES-NIE", mp => mp.Add(new PatternUnit(P.Single().WithShape("X9999999X"))));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddItFiscalCode(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "it_fiscal", "IT_FISCAL_CODE");
            // RSSMRA80A01H501U -> XXXXXX99X99X999X
            spotter.NewPattern("IT-Fiscal-Standard", mp => mp.Add(new PatternUnit(P.Single().WithShape("XXXXXX99X99X999X"))));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddSgNric(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "sg_nric", "SG_NRIC_FIN");
            // S1234567D -> X9999999X
            spotter.NewPattern("SG-NRIC", mp => mp.Add(new PatternUnit(P.Single().WithShape("X9999999X"))));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddAuAbn(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "au_abn", "AU_ABN");
            // 11 digits
            spotter.NewPattern("AU-ABN-Single", mp => mp.Add(new PatternUnit(P.Single().IsNumeric().WithLength(11, 11))));
            // Formatted: 51 824 753 556 -> 2, 3, 3, 3
            spotter.NewPattern("AU-ABN-Formatted", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3))
            ));
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddAuTfn(this PresidioAnalyzer analyzer)
        {
            var spotter = new PatternSpotter(analyzer.Language, 0, "au_tfn", "AU_TFN");
            // 8 or 9 digits
            spotter.NewPattern("AU-TFN-Single", mp => mp.Add(new PatternUnit(P.Single().IsNumeric().WithLength(8, 9))));
            // Formatted: 123 456 789 -> 3, 3, 3
            spotter.NewPattern("AU-TFN-Formatted-9", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3))
            ));
             // Formatted: 123 45 678 -> 3, 2, 3 (Not strictly standard but possible?)
             // Keeping it simple with 3-3-3 for now.
            return analyzer.AddRecognizer(spotter);
        }

        public static PresidioAnalyzer AddAllRecognizers(this PresidioAnalyzer analyzer)
        {
            return analyzer
                .AddEmail()
                .AddUrl()
                .AddAllPhones()
                .AddCreditCard()
                .AddIp()
                .AddUsSsn()
                .AddUsPassport()
                .AddIban()
                .AddUsItin()
                .AddCrypto()
                .AddUsDriverLicense()
                .AddUsBankNumber()
                .AddUkNhs()
                .AddEsNif()
                .AddItFiscalCode()
                .AddSgNric()
                .AddAuAbn()
                .AddAuTfn();
        }
    }
}
