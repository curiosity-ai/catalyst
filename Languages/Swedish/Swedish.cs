
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Swedish
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Swedish, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Swedish, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Swedish, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Swedish, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Swedish, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Swedish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Swedish, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Swedish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Swedish, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Swedish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Swedish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Swedish).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Swedish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Swedish, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Swedish, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Swedish, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Swedish, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
