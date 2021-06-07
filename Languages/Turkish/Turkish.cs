
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Turkish
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Turkish, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Turkish, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Turkish, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Turkish, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Turkish, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Turkish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Turkish, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Turkish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Turkish, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Turkish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Turkish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Turkish).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Turkish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Turkish, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Turkish, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Turkish, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Turkish, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
