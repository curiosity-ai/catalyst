using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static Catalyst.WordNet;


namespace Catalyst
{
    public static class WordNetExtensions
    {
        [Obsolete("Use WordNetSynonymsAsync")]
        public static IEnumerable<(string word, int lexId)> WordNetSynonyms(this IToken token, Language language)
        {
            if (language != Language.English) //TODO: See if we can add more languages from other sources similar to WordNet
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

        public static async Task<IEnumerable<(string Word, int LexId)>> WordNetSynonymsAsync(this IToken token, Language language)
        {
            if (language != Language.English)
            {
                var mapping = await WordNetMapping.FromStoreAsync(language, 0, "");
                if (!mapping.Loaded)
                {
                    throw new Exception($"WordNet has not been loaded for the language '{language}' or is not implemented.");
                }

                return mapping.GetSynonyms(token.Value, token.POS);
            }

            return token.POS switch
            {
                PartOfSpeech.ADJ => WordNet.Nouns.GetSynonyms(token.Value),
                PartOfSpeech.ADV => WordNet.Adverbs.GetSynonyms(token.Value),
                PartOfSpeech.PROPN or PartOfSpeech.NOUN => WordNet.Nouns.GetSynonyms(token.Value),
                PartOfSpeech.VERB => WordNet.Verbs.GetSynonyms(token.Value),
                _ => Enumerable.Empty<(string, int)>(),
            };
        }

        [Obsolete("Use WordNetPointersAsync")]
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

        public static async Task<IEnumerable<WordNetPointers>> WordNetPointersAsync(this IToken token, Language language)
        {
            if (language != Language.English)
            {
                var mapping = await WordNetMapping.FromStoreAsync(language, 0, "");
                if (!mapping.Loaded)
                {
                    throw new Exception($"WordNet has not been loaded for the language '{language}' or is not implemented.");
                }

                return mapping.GetPointers(token.Value);
            }

            var pointers = token.POS switch
            {
                PartOfSpeech.ADJ => WordNet.Nouns.GetPointers(token.Value),
                PartOfSpeech.ADV => WordNet.Adverbs.GetPointers(token.Value),
                PartOfSpeech.PROPN or PartOfSpeech.NOUN => WordNet.Nouns.GetPointers(token.Value),
                PartOfSpeech.VERB => WordNet.Verbs.GetPointers(token.Value),
                _ => Enumerable.Empty<(string Word, PointerSymbol Symbol, PartOfSpeech PartOfSpeech, byte Source, byte Target)>(),
            };

            return from pointer in pointers
                   select new WordNetPointers(0, pointer.Symbol, pointer.PartOfSpeech, pointer.Source, pointer.Target)
                   {
                       SourceWord = pointer.Word
                   };
        }
    }
}
