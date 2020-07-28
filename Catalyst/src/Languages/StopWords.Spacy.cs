using Mosaik.Core;
using System.Collections.Generic;

namespace Catalyst
{
    public static partial class StopWords
    {
        public static partial class Spacy
        {
            private static readonly ReadOnlyHashSet<string> Empty = new ReadOnlyHashSet<string>(new HashSet<string>());

            // Stop-words used by Spacy: https://github.com/explosion/spaCy/tree/master/spacy/lang

            public static ReadOnlyHashSet<string> For(Language lang)
            {
                return lang switch
                {
                    Language.Any        => Empty,
                    Language.English    => English,
                    Language.French     => French,
                    Language.German     => German,
                    Language.Italian    => Italian,
                    Language.Spanish    => Spanish,
                    Language.Portuguese => Portuguese,
                    _                   => Empty
                };
            }
        }
    }
}