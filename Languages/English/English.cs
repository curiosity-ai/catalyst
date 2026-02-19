using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Catalyst;
using Mosaik.Core;
using static Catalyst.WordNet;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.English, 0).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "tagger.bin", async (s) => { var a = new AveragePerceptronTagger(Language.English, 0, ""); await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.English, 0).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "parser.bin", async (s) => { var a = new AveragePerceptronDependencyParser(Language.English, 0, ""); await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "sentence-detector.bin", async (s) => { var a = new SentenceDetector(Language.English, 0, ""); await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0, "lower").GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.English, 0, ""); await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0, "upper").GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.English, 0, ""); await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.English, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "wikiner.bin", async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.English, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }); await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.English, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.English, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.English, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.English, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe: true));
        }

        public static async Task<IWordNet> GetWordNetAsync()
        {
            return new WordNetPassthrough();
        }

        private class WordNetPassthrough : IWordNet
        {
            public IWordNetData Nouns => WordNet.Nouns;
            public IWordNetData Verbs => WordNet.Verbs;
            public IWordNetData Adjectives => WordNet.Adjectives;
            public IWordNetData Adverbs => WordNet.Adverbs;

            public IEnumerable<WordNetPointers> GetPointers(WordNet.WordNetTerm term)
            {
                var data = GetDataImplementation(term.PartOfSpeech);
                return data.GetPointers(term);
            }

            /// <inheritdoc/>
            public IEnumerable<WordNetTerm> GetTerms(string word, PartOfSpeech partOfSpeech = PartOfSpeech.NONE)
            {
                IEnumerable<WordNetData> data;
                if (partOfSpeech == PartOfSpeech.NONE)
                {
                    data = new[] {
                        WordNet.Nouns,
                        WordNet.Verbs,
                        WordNet.Adjectives,
                        WordNet.Adverbs
                    };
                }
                else
                {
                    data = new[] {
                        GetDataImplementation(partOfSpeech)
                    };
                }

                foreach (var item in data)
                {
                    foreach (var term in item.GetTerms(word))
                    {
                        yield return term;
                    }
                }
            }

            /// <inheritdoc/>
            public IWordNetData GetData(PartOfSpeech partOfSpeech)
            {
                return this.GetDataImplementation(partOfSpeech);
            }

            private WordNetData GetDataImplementation(PartOfSpeech partOfSpeech)
            {
                return (WordNetData)(partOfSpeech switch
                {
                    PartOfSpeech.ADJ => Adjectives,
                    PartOfSpeech.ADV => Adverbs,
                    PartOfSpeech.NOUN => Nouns,
                    PartOfSpeech.VERB => Verbs,
                    _ => null,
                });
            }

            /// <inheritdoc/>
            public IEnumerable<string> GetWords(WordNetTerm term)
            {
                var data = GetDataImplementation(term.PartOfSpeech);
                return data.GetWords(term);
            }
        }

    }

}
