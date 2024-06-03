
using System.Collections.Generic;
using static Catalyst.WordNet;

namespace Catalyst
{
    public interface IWordNet
    {
        public IWordNetData Nouns { get; }
        public IWordNetData Verbs { get; }
        public IWordNetData Adjectives { get; }
        public IWordNetData Adverbs { get; }

        /// <summary>
        /// Gets the <see cref="IWordNetData"/> for this Part of Speech.
        /// </summary>
        /// <param name="PartOfSpeech"></param>
        /// <returns></returns>
        public IWordNetData GetData(PartOfSpeech partOfSpeech);

        /// <summary>
        /// Gets the pointers starting from a <see cref="WordNetTerm"/>.
        /// </summary>
        /// <param name="term"></param>
        /// <returns></returns>
        public IEnumerable<WordNetPointers> GetPointers(WordNetTerm term);

        /// <summary>
        /// Gets the terms for a (localized) word.
        /// </summary>
        /// <param name="word"></param>
        /// <param name="partOfSpeech"></param>
        /// <returns></returns>
        public IEnumerable<WordNetTerm> GetTerms(string word, PartOfSpeech partOfSpeech = PartOfSpeech.NONE);
    }
}
