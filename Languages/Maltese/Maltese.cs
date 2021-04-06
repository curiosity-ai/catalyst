
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Maltese
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Maltese, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Maltese, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Maltese, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Maltese, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Maltese, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Maltese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Maltese, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Maltese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Maltese, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Maltese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Maltese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Maltese).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Maltese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Maltese, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Maltese, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Maltese, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Maltese, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
