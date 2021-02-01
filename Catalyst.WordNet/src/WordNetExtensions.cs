using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using UID;


namespace Catalyst
{
    public static class WordNetExtensions
    {
        public static IEnumerable<(string word, int lexId)> WordNetSynonyms(this IToken token, Language language)
        {
            if(language != Language.English) //TODO: See if we can add more languages from other sources similar to WordNet
            {
                throw new Exception("Only english is supported");
            }

            switch (token.POS)
            {
                case PartOfSpeech.ADJ:
                    return WordNet.Nouns.GetSynonyms(token.Value);
                case PartOfSpeech.ADV:
                    return WordNet.Adverbs.GetSynonyms(token.Value);
                case PartOfSpeech.PROPN:
                case PartOfSpeech.NOUN:
                    return WordNet.Nouns.GetSynonyms(token.Value);
                case PartOfSpeech.VERB:
                    return WordNet.Verbs.GetSynonyms(token.Value);
                default:
                    return Enumerable.Empty<(string, int)>();
            }
        }

        public static IEnumerable<(string Word, WordNet.PointerSymbol Symbol, PartOfSpeech PartOfSpeech, byte Source, byte Target)> WordNetPointers(this IToken token, Language language)
        {
            if (language != Language.English) //TODO: See if we can add more languages from other sources similar to WordNet
            {
                throw new Exception("Only english is supported");
            }

            switch (token.POS)
            {
                case PartOfSpeech.ADJ:
                    return WordNet.Nouns.GetPointers(token.Value);
                case PartOfSpeech.ADV:
                    return WordNet.Adverbs.GetPointers(token.Value);
                case PartOfSpeech.PROPN:
                case PartOfSpeech.NOUN:
                    return WordNet.Nouns.GetPointers(token.Value);
                case PartOfSpeech.VERB:
                    return WordNet.Verbs.GetPointers(token.Value);
                default:
                    return Enumerable.Empty<(string, WordNet.PointerSymbol, PartOfSpeech, byte, byte)>();
            }
        }
    }
}
