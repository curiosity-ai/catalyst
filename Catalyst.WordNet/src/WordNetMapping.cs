using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mosaik.Core;
using static Catalyst.WordNet;

namespace Catalyst
{
    public class WordNetMapping : StorableObjectV2<WordNetMapping, WordNetMappingProperties>, IWordNet
    {
        public IWordNetData Adjectives { get; private set; }
        public IWordNetData Adverbs { get; private set; }
        public IWordNetData Nouns { get; private set; }
        public IWordNetData Verbs { get; private set; }

        /// <summary>
        /// Gets whether the data has been loaded for this language.
        /// </summary>
        public bool Loaded
        {
            get
            {
                return Data?.Loaded ?? false;
            }
        }

        public new static async Task<WordNetMapping> FromStoreAsync(Language language, int version, string tag)
        {
            var mapping = new WordNetMapping(language, version, tag);
            await mapping.LoadDataAsync();
            return mapping;
        }

        /// <summary>
        /// Reads the WordNet mapping to allow using localized WordNets
        /// </summary>
        /// <param name="stream">Resource stream containing the word net mapping</param>
        public void ReadData(Stream stream)
        {
            Debug.Assert(stream != null);
            Data.Mapping = new Dictionary<string, List<(int, PartOfSpeech)>>();
            Data.InverseMapping = new Dictionary<(string, int), HashSet<string>>();

            foreach (var line in Read(stream))
            {
                // line example:
                // 02827950-a	nld:lemma	hydro-elektrisch
                if (line.StartsWith("#") || string.IsNullOrEmpty(line))
                {
                    continue;
                }

                var offset = int.Parse(line[.."01066881".Length]);
                var partOfSpeech = line.Substring("01066881-".Length, 1)[0] switch
                {
                    'n' => PartOfSpeech.NOUN,
                    'v' => PartOfSpeech.VERB,
                    'a' => PartOfSpeech.ADJ,
                    's' => PartOfSpeech.ADJ,
                    'r' => PartOfSpeech.ADV,
                    _ => PartOfSpeech.X,
                };
                var cells = line.Split('\t');
                var lemma = cells[2];

                if (Data.Mapping.ContainsKey(lemma))
                {
                    Data.Mapping[lemma].Add((offset, partOfSpeech));
                }
                else
                {
                    Data.Mapping[lemma] = new List<(int, PartOfSpeech)> { (offset, partOfSpeech) };
                }

                foreach (var mappedLemma in GetMapping(lemma, partOfSpeech))
                {
                    if (Data.InverseMapping.ContainsKey(mappedLemma))
                    {
                        Data.InverseMapping[mappedLemma].Add(lemma);
                    }
                    else
                    {
                        Data.InverseMapping[mappedLemma] = new HashSet<string> { lemma };
                    }
                }
            }

            Data.Loaded = true;
        }

        public WordNetMapping(Language language, int version, string tag = "", bool compress = true) : base(language, version, tag, compress)
        {
            Adjectives = new WordNetDataMapping(this, PartOfSpeech.ADJ, WordNet.Adjectives);
            Adverbs = new WordNetDataMapping(this, PartOfSpeech.ADV, WordNet.Adverbs);
            Nouns = new WordNetDataMapping(this, PartOfSpeech.NOUN, WordNet.Nouns);
            Verbs = new WordNetDataMapping(this, PartOfSpeech.VERB, WordNet.Verbs);
        }

        /// <inheritdoc/>
        public IWordNetData GetData(PartOfSpeech partOfSpeech)
        {
            return this.GetDataMapping(partOfSpeech);
        }

        private WordNetDataMapping GetDataMapping(PartOfSpeech partOfSpeech)
        {
            return (WordNetDataMapping)(partOfSpeech switch
            {
                PartOfSpeech.ADJ => Adjectives,
                PartOfSpeech.ADV => Adverbs,
                PartOfSpeech.NOUN => Nouns,
                PartOfSpeech.VERB => Verbs,
                _ => null,
            });
        }

        /// <summary>
        /// Gets the mapping of the lemma to a WordNet entry
        /// </summary>
        /// <param name="word">localized word</param>
        /// <param name="partOfSpeech"></param>
        /// <returns></returns>
        public IEnumerable<(string Word, int LexId)> GetMapping(string word, PartOfSpeech partOfSpeech)
        {
            WordNetDataMapping posMapping = GetDataMapping(partOfSpeech);
            if (Data.Mapping.ContainsKey(word))
            {
                foreach ((var offset, var itemPoS) in Data.Mapping[word])
                {
                    if (itemPoS == partOfSpeech)
                    {
                        var term = posMapping.Terms[offset];
                        var lexId = term.LexID;
                        yield return (posMapping.GetWordFromCache(term.WordStart, term.WordLength), lexId);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the localized words from the original WordNet word
        /// </summary>
        /// <param name="word"></param>
        /// <param name="lexId"></param>
        /// <returns></returns>
        internal IEnumerable<string> GetInverseMapping(string word, int lexId)
        {
            var key = (word, lexId);
            if (Data.InverseMapping.ContainsKey(key))
            {
                return Data.InverseMapping[key];
            }

            return Enumerable.Empty<string>();
        }

        /// <summary>
        /// Gets the synonyms and their lexical IDs (in the original file)
        /// </summary>
        /// <param name="word"></param>
        /// <param name="partOfSpeech"></param>
        /// <returns></returns>
        public IEnumerable<(string Word, int LexId)> GetSynonyms(string word, PartOfSpeech partOfSpeech)
        {
            var posMapping = GetDataMapping(partOfSpeech);
            var synonyms = new HashSet<(string, int)>();
            foreach ((var mappedWord, var lexId) in GetMapping(word, partOfSpeech))
            {
                foreach ((var synonymWord, var synonymLexId) in posMapping.data.GetSynonyms(mappedWord, lexId))
                {
                    foreach (var inverted in GetInverseMapping(synonymWord, synonymLexId))
                    {
                        synonyms.Add((inverted, synonymLexId));
                    }
                }
            }

            return synonyms;
        }

        /// <inheritdoc />
        public IEnumerable<WordNetTerm> GetTerms(string word, PartOfSpeech partOfSpeech = PartOfSpeech.NONE)
        {
            if (Data.Mapping.ContainsKey(word))
            {
                foreach (var (offset, sourcePartOfSpeech) in Data.Mapping[word])
                {
                    if (partOfSpeech != PartOfSpeech.NONE && partOfSpeech != sourcePartOfSpeech)
                    {
                        continue;
                    }

                    WordNetDataMapping posMapping = GetDataMapping(sourcePartOfSpeech);
                    if (posMapping == null)
                    {
                        continue;
                    }

                    yield return posMapping.Terms[offset];
                }
            }
        }

        /// <summary>
        /// Gets the WordNet words for a term, use <see cref="GetInverseMapping"/> to localize.
        /// </summary>
        /// <param name="term"></param>
        /// <returns></returns>
        public IEnumerable<string> GetWords(WordNetTerm term)
        {
            var data = GetDataMapping(term.PartOfSpeech);
            var word = data.GetWordFromCache(term.WordStart, term.WordLength);
            return this.GetInverseMapping(word, term.LexID);
        }

        /// <inheritdoc/>        
        public IEnumerable<WordNetPointers> GetPointers(WordNetTerm term)
        {
            WordNetDataMapping posMapping = GetDataMapping(term.PartOfSpeech);
            if (posMapping == null)
            {
                yield break;
            }

            foreach ((string word, WordNetTerm target, WordNetPointers pointer) in posMapping.GetPointers(term))
            {
                foreach (var targetWord in GetWords(target))
                {
                    yield return new WordNetPointers(target.Offset, pointer.Symbol, pointer.PartOfSpeech, pointer.Source, pointer.Target)
                    {
                        SourceWord = word,
                        TargetWord = targetWord
                    };
                }
            }
        }

        public IEnumerable<WordNetPointers> GetPointers(string word, int lexId = -1)
        {
            if (Data.Mapping.ContainsKey(word))
            {
                foreach (var (offset, sourcePartOfSpeech) in lexId == -1 ? Data.Mapping[word] : new List<(int, PartOfSpeech)> { Data.Mapping[word][lexId] })
                {
                    WordNetDataMapping posMapping = GetDataMapping(sourcePartOfSpeech);
                    if (posMapping == null)
                    {
                        continue;
                    }

                    var source = posMapping.Terms[offset];
                    foreach (var pointer in GetPointers(source))
                    {
                        yield return pointer;
                    }
                }
            }
        }

        private IEnumerable<string> Read(Stream resourceStream)
        {
            using var reader = new StreamReader(resourceStream, Encoding.UTF8);
            while (reader.Peek() >= 0)
            {
                var line = reader.ReadLine();
                yield return line;
            }
        }
    }
}
