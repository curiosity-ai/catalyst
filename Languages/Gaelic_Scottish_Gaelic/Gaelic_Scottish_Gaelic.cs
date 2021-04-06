
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Gaelic_Scottish_Gaelic
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Gaelic_Scottish_Gaelic, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Gaelic_Scottish_Gaelic, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Gaelic_Scottish_Gaelic, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Gaelic_Scottish_Gaelic, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Gaelic_Scottish_Gaelic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Gaelic_Scottish_Gaelic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Gaelic_Scottish_Gaelic).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Gaelic_Scottish_Gaelic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Gaelic_Scottish_Gaelic, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Gaelic_Scottish_Gaelic, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Gaelic_Scottish_Gaelic, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Gaelic_Scottish_Gaelic, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
