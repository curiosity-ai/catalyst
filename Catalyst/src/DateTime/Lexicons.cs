using System;
using Mosaik.Core;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>Maps a <see cref="Language"/> onto the vocabulary the parser should use for it.</summary>
    public static class Lexicons
    {
        public static bool IsSupported(Language language) => language switch
        {
            Language.English or Language.German or Language.French or Language.Spanish
                or Language.Portuguese or Language.Italian or Language.Dutch => true,
            _                                                                => false,
        };

        public static Lexicon For(Language language, bool useUsEnglishForEnglish = false) => language switch
        {
            Language.English    => EnglishLexicon.Get(dayMonthOrder: !useUsEnglishForEnglish),
            Language.German     => GermanLexicon.Get(),
            Language.French     => FrenchLexicon.Get(),
            Language.Spanish    => SpanishLexicon.Get(),
            Language.Portuguese => PortugueseLexicon.Get(),
            Language.Italian    => ItalianLexicon.Get(),
            Language.Dutch      => DutchLexicon.Get(),
            _                   => throw new NotSupportedException($"No date/time vocabulary for {language}"),
        };
    }
}
