
using System.Collections.Generic;
using static Catalyst.WordNet;

namespace Catalyst
{
    public interface IWordNetData
    {
        /// <summary>
        /// Gets the synonyms and their lexical IDs (in the original file)
        /// </summary>
        /// <param name="word"></param>
        /// <param name="lexId"></param>
        /// <returns></returns>
        IEnumerable<(string Word, int LexId)> GetSynonyms(string word, int lexId = -1);

        /// <summary>
        /// Gets the pointers for the (localized) word
        /// </summary>
        /// <param name="word"></param>
        /// <param name="lexId">Lexical ID as used in the original WordNet</param>
        /// <returns></returns>
        IEnumerable<(int Offset, string Word, WordNet.PointerSymbol Symbol, PartOfSpeech PartOfSpeech, byte Source, byte Target)> GetPointers(string word, int lexId = -1);

        /// <summary>
        /// Gets the <see cref="WordNetTerm"/> by its byte offset.
        /// </summary>
        /// <param name="offset"></param>
        /// <returns></returns>
        WordNetTerm GetTerm(int offset);

        /// <summary>
        /// Gets the (localized) words for this term. For English this is only one word.
        /// </summary>
        IEnumerable<string> GetWords(WordNetTerm term);
    }
}
