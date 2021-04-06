
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Chinese
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Chinese, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Chinese, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Chinese, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Chinese, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Chinese, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Chinese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Chinese, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Chinese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Chinese, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Chinese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Chinese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Chinese).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Chinese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Chinese, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Chinese, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Chinese, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Chinese, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
