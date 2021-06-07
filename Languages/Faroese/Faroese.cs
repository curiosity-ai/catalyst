
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Faroese
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Faroese, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Faroese, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Faroese, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Faroese, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Faroese, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Faroese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Faroese, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Faroese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Faroese, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Faroese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Faroese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Faroese).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Faroese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Faroese, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Faroese, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Faroese, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Faroese, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
