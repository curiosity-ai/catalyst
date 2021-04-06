
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Slovak
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Slovak, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Slovak, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Slovak, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Slovak, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Slovak, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Slovak, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Slovak, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Slovak, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Slovak, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Slovak, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Slovak, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Slovak).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Slovak, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Slovak, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Slovak, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Slovak, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Slovak, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
