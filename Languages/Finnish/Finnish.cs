
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Finnish
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Finnish, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Finnish, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Finnish, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Finnish, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Finnish, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Finnish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Finnish, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Finnish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Finnish, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Finnish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Finnish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Finnish).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Finnish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Finnish, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Finnish, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Finnish, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Finnish, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
