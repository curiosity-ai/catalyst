
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Urdu
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Urdu, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Urdu, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Urdu, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Urdu, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Urdu, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Urdu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Urdu, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Urdu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Urdu, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Urdu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Urdu, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Urdu).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Urdu, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Urdu, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Urdu, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Urdu, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Urdu, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
