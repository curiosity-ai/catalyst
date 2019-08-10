using Mosaik.Core;
using System.Collections.Generic;

namespace Catalyst
{
    public static partial class StopWords
    {
        public static partial class Spacy
        {
            private static ReadOnlyHashSet<string> Empty = new ReadOnlyHashSet<string>(new HashSet<string>());

            //Stop-words used by Spacy: https://github.com/explosion/spaCy/tree/master/spacy/lang

            public static ReadOnlyHashSet<string> For(Language lang)
            {
                switch (lang)
                {
                    case Language.Any:        return Empty;
                    case Language.English:    return English;
                    case Language.French:     return French;
                    case Language.German:     return German;
                    case Language.Italian:    return Italian;
                    case Language.Spanish:    return Spanish;
                    case Language.Portuguese: return Portuguese;
                    default:                  return Empty;
                }
            }
        }
    }
}