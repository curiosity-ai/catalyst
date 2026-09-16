using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>Maps a <see cref="Language"/> onto the vocabulary the parser should use for it.</summary>
    public static class Lexicons
    {
        public static bool IsSupported(Language language) => language switch
        {
            Language.English => true,
            _                => false,
        };

        public static Lexicon For(Language language, bool useUsEnglishForEnglish = false) => language switch
        {
            Language.English    => EnglishLexicon.Get(dayMonthOrder: !useUsEnglishForEnglish),
            _                   => throw new NotSupportedException($"No date/time vocabulary for {language}"),
        };
    }
}
