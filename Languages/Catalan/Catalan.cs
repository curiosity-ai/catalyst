
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Catalan
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Catalan, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Catalan, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Catalan, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Catalan, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Catalan, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Catalan, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Catalan, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Catalan, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Catalan, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Catalan, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Catalan, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Catalan).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Catalan, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Catalan, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Catalan, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Catalan, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Catalan, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
