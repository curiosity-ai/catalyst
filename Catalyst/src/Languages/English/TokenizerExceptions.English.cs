using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
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
                        Create(EnglishExceptions, "", "'ll|'re|'d|'cause|'em|'nuff|doin'|goin'|nothin'|ol'|somethin'", "will|are|had|because|them|enough|doing|going|nothing|old|something");
                        Create(EnglishExceptions, "", "can't|cant|shall've|won't|wont|ain't|aint", "can not|can not|shall have|will not|will not|is not|is not");
                        Create(EnglishExceptions, "", "and/or|o.k.", "and/or|ok");
                        Create(EnglishExceptions, "", "y'all|yall|ma'am|o'clock|oclock|how'd'y|not've|notve|cannot|gonna|gotta|let's|lets",
                                                      "you all|you all|madam|o'clock|o'clock|how do you|not have|not have|can not|going to|got to|let's|let's");
                        Create(EnglishExceptions, "", "a.m.|adm.|bros.|co.|corp.|d.c.|dr.|e.g.|gen.|gov.|i.e.|inc.|jr.|ltd.|md.|messrs.|mo.|mont.|mr.|mrs.|ms.|p.m.|ph.d.|rep.|rev.|sen.|st.|vs.|A.m.|D.c.|E.g.|I.e.|P.m.|Ph.D.",
                                                      "a.m.|adm.|bros.|co.|corp.|d.c.|dr.|e.g.|gen.|gov.|i.e.|inc.|jr.|ltd.|md.|messrs.|mo.|mont.|mr.|mrs.|ms.|p.m.|ph.d.|rep.|rev.|sen.|st.|vs.|A.m.|D.c.|E.g.|I.e.|P.m.|Ph.D.");

                    }
                    //TODO: Check if should add any of the contractions here: https://en.wiktionary.org/wiki/Category:English_contractions
                    //TODO: Add verbs in gerund having the ending -ing replaced by -in'  , such as lovin' ->loving
                    //TODO: Add numbers in this form: 14th
                }
            }
            return EnglishExceptions;
        }
    }
}
