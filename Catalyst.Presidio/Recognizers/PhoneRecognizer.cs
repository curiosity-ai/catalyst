using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using P = Catalyst.PatternUnitPrototype;

namespace Catalyst.Presidio
{
    public static class PhoneRecognizer
    {
        public static PresidioAnalyzer AddPhoneUS(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_us", "PHONE_US");
            // US / Canada
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

        public static PresidioAnalyzer AddPhoneUK(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_uk", "PHONE_UK");
            // UK
            // 07700 900077 (5, 6)
            phoneSpotter.NewPattern("Phone-UK-5-6", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(5, 5).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(6, 6))
            ));
            // 020 7946 0123 (3, 4, 4)
            phoneSpotter.NewPattern("Phone-UK-3-4-4", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            // +44 ...
            phoneSpotter.NewPattern("Phone-UK-Intl-2-4-6", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")), // +44
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(6, 6))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneDE(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_de", "PHONE_DE");
            // Germany
            // 0xx xxxxx
            phoneSpotter.NewPattern("Phone-DE-Area", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 5).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 8))
            ));
            // +49 ...
            phoneSpotter.NewPattern("Phone-DE-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 5)),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 8))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneFR(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_fr", "PHONE_FR");
            // France
            // 0x xx xx xx xx (2, 2, 2, 2, 2)
            phoneSpotter.NewPattern("Phone-FR-Local", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2))
            ));
            // +33 x xx ...
            phoneSpotter.NewPattern("Phone-FR-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(1, 1)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneBR(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_br", "PHONE_BR");
            // Brazil
            // (xx) xxxxx-xxxx
            phoneSpotter.NewPattern("Phone-BR-Parens", mp => mp.Add(
                new PatternUnit(P.Single().IsOpeningParenthesis()),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsClosingParenthesis()),
                new PatternUnit(P.Single().WithShape("99999-9999"))
            ));
             phoneSpotter.NewPattern("Phone-BR-Parens-Old", mp => mp.Add(
                new PatternUnit(P.Single().IsOpeningParenthesis()),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().IsClosingParenthesis()),
                new PatternUnit(P.Single().WithShape("9999-9999"))
            ));
            // +55 ...
            phoneSpotter.NewPattern("Phone-BR-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2)),
                new PatternUnit(P.Single().WithShape("99999-9999"))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneIN(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_in", "PHONE_IN");
            // India
            // +91 xxxxx xxxxx
            phoneSpotter.NewPattern("Phone-IN-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(5, 5)),
                new PatternUnit(P.Single().IsNumeric().WithLength(5, 5))
            ));
            // 0xxxxx xxxxx
            phoneSpotter.NewPattern("Phone-IN-Local", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 5).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(5, 8))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneCN(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_cn", "PHONE_CN");
            // China
            // +86 1xx xxxx xxxx
            phoneSpotter.NewPattern("Phone-CN-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3).WithPrefix("1")),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddPhoneAU(this PresidioAnalyzer analyzer)
        {
            var phoneSpotter = new PatternSpotter(analyzer.Language, 0, "phone_au", "PHONE_AU");
            // Australia
            // 0x xxxx xxxx
            phoneSpotter.NewPattern("Phone-AU-Local", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(2, 2).WithPrefix("0")),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            // +61 x ...
            phoneSpotter.NewPattern("Phone-AU-Intl", mp => mp.Add(
                new PatternUnit(P.Single().WithShape("+99")),
                new PatternUnit(P.Single().IsNumeric().WithLength(1, 1)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            return analyzer.AddRecognizer(phoneSpotter);
        }

        public static PresidioAnalyzer AddAllPhones(this PresidioAnalyzer analyzer)
        {
            return analyzer
                .AddPhoneUS()
                .AddPhoneUK()
                .AddPhoneDE()
                .AddPhoneFR()
                .AddPhoneBR()
                .AddPhoneIN()
                .AddPhoneCN()
                .AddPhoneAU();
        }
    }
}
