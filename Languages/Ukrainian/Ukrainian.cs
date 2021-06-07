
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Ukrainian
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Ukrainian, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Ukrainian, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Ukrainian, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Ukrainian, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Ukrainian, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Ukrainian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Ukrainian, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Ukrainian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Ukrainian, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Ukrainian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Ukrainian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Ukrainian).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Ukrainian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Ukrainian, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Ukrainian, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Ukrainian, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Ukrainian, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
