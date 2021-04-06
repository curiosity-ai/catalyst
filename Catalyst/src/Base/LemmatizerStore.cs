using Mosaik.Core;
using System.Collections.Concurrent;

namespace Catalyst
{
    public static class LemmatizerStore
    {
        private static ILemmatizer _missingLemmatizer = new MissingLemmatizer();

        private static ConcurrentDictionary<Language, ILemmatizer> _perLanguage = new ConcurrentDictionary<Language, ILemmatizer>();
        public static void Register(Language language, ILemmatizer lemmatizer) => _perLanguage[language] = lemmatizer;
        public static ILemmatizer Get(Language language) => _perLanguage.TryGetValue(language, out var lemmatizer) ? lemmatizer : _missingLemmatizer;
    }
}