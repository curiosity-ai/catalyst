
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Marathi
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Marathi, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Marathi, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Marathi, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Marathi, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Marathi, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Marathi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Marathi, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Marathi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Marathi, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Marathi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Marathi, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Marathi).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Marathi, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Marathi, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Marathi, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Marathi, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Marathi, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
