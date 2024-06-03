using System.Collections.Generic;
using static Catalyst.WordNet;

namespace Catalyst
{
    /// <summary>
    /// Maps the <see cref="WordNetData"/> to another language using the passed <see cref="WordNetMapping"/>.
    /// </summary>
    public class WordNetDataMapping : IWordNetData
    {
        /// <summary>
        /// Key is term offset
        /// </summary>
        internal Dictionary<int, WordNetTerm> Terms { get; set; }
        private readonly WordNetPointers[] Pointers;
        private readonly WordNetMapping mapping;
        private readonly PartOfSpeech partOfSpeech;
        public WordNetData data;

        /// <param name="mapping"></param>
        /// <param name="wordNetData"></param>
        public WordNetDataMapping(WordNetMapping mapping, PartOfSpeech partOfSpeech, WordNetData wordNetData)
        {
            this.partOfSpeech = partOfSpeech;
            Terms = wordNetData.Terms;
            Pointers = wordNetData.Pointers;
            this.mapping = mapping;
            data = wordNetData;
        }

        /// <inheritdoc/>
        public IEnumerable<(string Word, int LexId)> GetSynonyms(string word, int lexId = -1)
        {
            foreach (var (synonymWord, synonymLexId) in this.mapping.GetSynonyms(word, this.partOfSpeech))
            {
                if (lexId == -1 || synonymLexId == lexId)
                {
                    yield return (synonymWord, synonymLexId);
                }
            }
        }

        /// <inheritdoc/>
        public IEnumerable<(int Offset, string Word, WordNet.PointerSymbol Symbol, PartOfSpeech PartOfSpeech, byte Source, byte Target)> GetPointers(string word, int lexId = -1)
        {
            foreach (var term in this.mapping.GetTerms(word, partOfSpeech))
            {
                if (lexId == -1 || term.LexID == lexId)
                {
                    foreach (var (word2, _, pointer) in this.GetPointers(term))
                    {
                        yield return (pointer.Offset, word2, pointer.Symbol, pointer.PartOfSpeech, pointer.Source, pointer.Target);
                    }
                }
            }
        }

        internal IEnumerable<(string Word, WordNetTerm Term, WordNetPointers Pointer)> GetPointers(WordNetTerm term)
        {
            for (int i = term.PointersStart; i < term.PointersStart + term.PointersLength; i++)
            {
                var pointer = Pointers[i];
                var otherData = (WordNetDataMapping)(pointer.PartOfSpeech switch
                {
                    PartOfSpeech.NOUN => mapping.Nouns,
                    PartOfSpeech.VERB => mapping.Verbs,
                    PartOfSpeech.ADJ => mapping.Adjectives,
                    PartOfSpeech.ADV => mapping.Adverbs,
                    _ => null,
                });

                if (otherData == null)
                {
                    continue;
                }

                var offset = pointer.Offset;
                var otherTerm = otherData.Terms[offset];
                foreach (var inverted in mapping.GetInverseMapping(otherData.GetWordFromCache(otherTerm.WordStart, otherTerm.WordLength), otherTerm.LexID))
                {
                    yield return (inverted, otherTerm, pointer);
                }
            }
        }

        /// <inheritdoc/>
        public WordNetTerm GetTerm(int offset)
        {
            return this.Terms[offset];
        }

        internal string GetWordFromCache(int start, int len)
        {
            return data.GetWordFromCache(start, len);
        }

        /// <inheritdoc/>
        public IEnumerable<string> GetWords(WordNetTerm term)
        {
            var word = data.GetWordFromCache(term.WordStart, term.WordLength);
            return this.mapping.GetInverseMapping(word, term.LexID);
        }
    }
}