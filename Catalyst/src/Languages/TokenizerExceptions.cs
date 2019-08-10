using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        public static Dictionary<int, TokenizationException> GetExceptions(Language language)
        {
            switch (language)
            {
                case Language.English: { return GetEnglishExceptions(); }
                default: { return new Dictionary<int, TokenizationException>(); }
            }
        }

        private static Dictionary<int, TokenizationException> BaseExceptions()
        {
            var exceptions = new Dictionary<int, TokenizationException>();
            foreach (var emoji in new string[] { ":)", ":-)", ":))", ":-))", ":)))", ":-)))", "(:", "(-:", "=)", "(=", "\")", ":]", ":-]", "[:", "[-:", ":o)", "(o:", ":}", ":-}", "8)", "8-)", "(-8", ";)", ";-)", "(;", "(-;", ":(", ":-(", ":((", ":-((", ":(((", ":-(((", "):", ")-:", "=(", ">:(", ":')", ":'-)", ":'(", ":'-(", ":/", ":-/", "=/", "=|", ":|", ":-|", ":1", ":P", ":-P", ":p", ":-p", ":O", ":-O", ":o", ":-o", ":0", ":-0", ":()", ">:o", ":*", ":-*", ":3", ":-3", "=3", ":>", ":->", ":X", ":-X", ":x", ":-x", ":D", ":-D", ";D", ";-D", "=D", "xD", "XD", "xDD", "XDD", "8D", "8-D", "^_^", "^__^", "^___^", ">.<", ">.>", "<.<", "._.", ";_;", "-_-", "-__-", "v.v", "V.V", "v_v", "V_V", "o_o", "o_O", "O_o", "O_O", "0_o", "o_0", "0_0", "o.O", "O.o", "O.O", "o.o", "0.0", "o.0", "0.o", "@_@", "<3", "<33", "<333", "</3", "(^_^)", "(-_-)", "(._.)", "(>_<)", "(*_*)", "(¬_¬)", "ಠ_ಠ", "ಠ︵ಠ", "(ಠ_ಠ)", "¯\\(ツ)/¯", "(╯°□°）╯︵┻━┻", "><(((*>" })
            {
                exceptions[emoji.CaseSensitiveHash32()] = new TokenizationException(new string[] { emoji });
            }
            return exceptions;
        }

        private static void Create(Dictionary<int, TokenizationException> englishExceptions, string part1, string part2, string replacements)
        {
            var separator = new char[] { '|' };

            var o1 = part1.Split(separator);
            var o2 = part2.Split(separator);
            var or = replacements.Split(separator);

            //Starts from lower-case, and composes later the other cases
            var p1 = part1.ToLowerInvariant().Split(separator);
            var p2 = part2.ToLowerInvariant().Split(separator);
            var r = replacements.ToLowerInvariant().Split(separator);

            if (p2.Length != r.Length)
            {
                throw new InvalidOperationException();
            }

            for (int i = 0; i < p1.Length; i++)
            {
                for (int j = 0; j < p2.Length; j++)
                {
                    var tk = (p1[i] + p2[j]);
                    var otk = (o1[i] + o2[j]);

                    var rep = new List<string>();
                    if (p1[i].Length > 0) { rep.Add(p1[i]); }
                    rep.AddRange(r[j].Split(new char[] { ' ' }));
                    //lower case
                    englishExceptions[tk.ToLowerInvariant().CaseSensitiveHash32()] = new TokenizationException(rep.ToArray());

                    //UPPER CASE
                    englishExceptions[tk.ToUpperInvariant().CaseSensitiveHash32()] = new TokenizationException(rep.Select(t => t.ToUpperInvariant()).ToArray());

                    //Title Case
                    if (tk != tk.ToTitleCase())
                    {
                        rep[0] = rep[0].ToTitleCase();
                        englishExceptions[tk.ToTitleCase().CaseSensitiveHash32()] = new TokenizationException(rep.ToArray());
                    }

                    if (tk != otk)
                    {
                        rep.Clear();
                        if (o1[i].Length > 0) { rep.Add(o1[i]); }
                        rep.AddRange(or[j].Split(new char[] { ' ' }));
                        englishExceptions[otk.CaseSensitiveHash32()] = new TokenizationException(rep.ToArray());
                    }
                }
            }
        }
    }
}