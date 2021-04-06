
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Japanese
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Japanese, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Japanese, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Japanese, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Japanese, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Japanese, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Japanese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Japanese, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Japanese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Japanese, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Japanese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Japanese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Japanese).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Japanese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Japanese, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Japanese, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Japanese, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Japanese, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
