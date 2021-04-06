
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Icelandic
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Icelandic, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Icelandic, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Icelandic, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Icelandic, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Icelandic, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Icelandic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Icelandic, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Icelandic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Icelandic, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Icelandic, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Icelandic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Icelandic).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Icelandic, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Icelandic, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Icelandic, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Icelandic, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Icelandic, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
