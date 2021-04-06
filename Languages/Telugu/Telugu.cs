
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Telugu
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Telugu, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Telugu, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Telugu, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Telugu, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Telugu, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Telugu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Telugu, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Telugu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Telugu, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Telugu, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Telugu, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Telugu).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Telugu, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Telugu, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Telugu, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Telugu, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Telugu, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
