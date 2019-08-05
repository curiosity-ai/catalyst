using UID;
using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst.Models
{
    [MessagePackObject]
    public struct TokenizationException
    {
        [Key(0)] public string[] Replacements;

        [SerializationConstructor]
        public TokenizationException(string[] replacements) { Replacements = replacements; }
    }

    public static class TokenizerExceptions
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

        private static object _lockEnglishExceptions = new object();
        private static Dictionary<int, TokenizationException> EnglishExceptions;

        private static Dictionary<int, TokenizationException> GetEnglishExceptions()
        {
            if (EnglishExceptions is null)
            {
                lock (_lockEnglishExceptions)
                {
                    if (EnglishExceptions is null)
                    {
                        EnglishExceptions = BaseExceptions();
                        //TODO: Check if should add any of the contractions here: https://en.wiktionary.org/wiki/Category:English_contractions
                        Create(EnglishExceptions, "i", "'m|'ma", "am|am going to");
                        Create(EnglishExceptions, "", "'m|'ma|n't|'s", "am|am going to|not|'s");
                        Create(EnglishExceptions, "", "shalln't|shan't", "shall not|shall not");
                        Create(EnglishExceptions, "", "I'dn't've|we'ven't|I'ven't|cou'dn't|wou'dn't|shou'dn't|she'sn't|he'sn't", "I would not have|we have not|I have not|could not|would not|should not|she has not|he has not");
                        Create(EnglishExceptions, "he|she|it", "'ll|'d|'ll've|'d've", "will|would|will have|would have");
                        Create(EnglishExceptions, "i|you|we|they", "'ve|'ll|'d|'ll've|'d've", "have|will|would|will have|would have");
                        Create(EnglishExceptions, "you|it|they", "ll", "will");
                        Create(EnglishExceptions, "you|they", "'re|re", "are|are");
                        Create(EnglishExceptions, "he|she|it", "'s", "is");
                        Create(EnglishExceptions, "who|what|when|where|why|how|there|that", "'ll|'d|'ve|'re|'s|'ll've|'d've", "will|would|have|are|is|will have|would have");
                        Create(EnglishExceptions, "could|do|does|did|had|may|might|must|need|ought|should|would", "n't|nt|n't've|ntve", "not|not|not have|not have");
                        Create(EnglishExceptions, "could|might|must|should|would", "'ve|ve", "have|have");
                        Create(EnglishExceptions, "is|are|was|were", "n't|nt", "not|not");
                        Create(EnglishExceptions, "0|1|2|3|4|5|6|7|8|9|10|11|12", "am|a.m|a.m.|am.|pm|p.m|p.m.|pm.", "a.m.|a.m.|a.m.|a.m.|p.m.|p.m.|p.m.|p.m.");
                        Create(EnglishExceptions, "", "'ll|'re|'d|'cause|'em|'nuff|doin'|goin'|nothin'|ol'|somethin'", "will|are|had|because|them|enough|doing|going|nothing|old|something");
                        Create(EnglishExceptions, "", "can't|cant|shall've|won't|wont|ain't|aint", "can not|can not|shall have|will not|will not|is not|is not");
                        Create(EnglishExceptions, "", "and/or|o.k.", "and/or|ok");
                        Create(EnglishExceptions, "", "y'all|yall|ma'am|o'clock|oclock|how'd'y|not've|notve|cannot|gonna|gotta|let's|lets",
                                                      "you all|you all|madam|o'clock|o'clock|how do you|not have|not have|can not|going to|got to|let's|let's");
                        Create(EnglishExceptions, "", "a.m.|adm.|bros.|co.|corp.|d.c.|dr.|e.g.|gen.|gov.|i.e.|inc.|jr.|ltd.|md.|messrs.|mo.|mont.|mr.|mrs.|ms.|p.m.|ph.d.|rep.|rev.|sen.|st.|vs.|A.m.|D.c.|E.g.|I.e.|P.m.|Ph.D.",
                                                      "a.m.|adm.|bros.|co.|corp.|d.c.|dr.|e.g.|gen.|gov.|i.e.|inc.|jr.|ltd.|md.|messrs.|mo.|mont.|mr.|mrs.|ms.|p.m.|ph.d.|rep.|rev.|sen.|st.|vs.|A.m.|D.c.|E.g.|I.e.|P.m.|Ph.D.");

                        //CreateCombinedExceptions(EnglishExceptions, "", "", "");
                    }
                    //TODO: Add verbs in gerund having the ending -ing replaced by -in'  , such as lovin' ->loving
                    //TODO: Add numbers in this form: 14th
                }
            }

            return EnglishExceptions;
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