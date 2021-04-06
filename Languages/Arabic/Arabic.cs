
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Arabic
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Arabic, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Arabic, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Arabic, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Arabic, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Arabic, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Arabic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Arabic, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Arabic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Arabic, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Arabic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Arabic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Arabic).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Arabic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Arabic, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Arabic, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Arabic, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Arabic, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
