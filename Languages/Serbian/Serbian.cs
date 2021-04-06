
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Serbian
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Serbian, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Serbian, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Serbian, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Serbian, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Serbian, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Serbian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Serbian, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Serbian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Serbian, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Serbian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Serbian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Serbian).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Serbian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Serbian, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Serbian, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Serbian, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Serbian, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
