using Mosaik.Core;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Catalyst
{
    public static partial class StopWords
    {
        //Stop-words used by Spacy: https://github.com/explosion/spaCy/tree/master/spacy/lang
        public static partial class Spacy
        {
            private static ReadOnlyHashSet<string> Empty = new ReadOnlyHashSet<string>(new HashSet<string>());
            
            private static ConcurrentDictionary<Language, ReadOnlyHashSet<string>> _perLanguage = new ConcurrentDictionary<Language, ReadOnlyHashSet<string>>();
            
            public static void Register(Language language, ReadOnlyHashSet<string> stopWords) => _perLanguage[language] = stopWords;

            public static ReadOnlyHashSet<string> For(Language lang) => _perLanguage.TryGetValue(lang, out var set) ? set : Empty;
        }
    }
}