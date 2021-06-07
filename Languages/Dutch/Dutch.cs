
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Dutch
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Dutch, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Dutch, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Dutch, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Dutch, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Dutch, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Dutch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Dutch, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Dutch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Dutch, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Dutch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Dutch, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Dutch).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Dutch, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Dutch, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Dutch, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Dutch, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Dutch, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
