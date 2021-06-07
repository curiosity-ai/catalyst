
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Polish
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Polish, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Polish, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Polish, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Polish, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Polish, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Polish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Polish, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Polish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Polish, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Polish, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Polish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Polish).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Polish, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Polish, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Polish, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Polish, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Polish, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
