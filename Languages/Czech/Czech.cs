
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Czech
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Czech, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Czech, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Czech, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Czech, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Czech, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Czech, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Czech, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Czech, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Czech, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Czech, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Czech, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Czech).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Czech, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Czech, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Czech, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Czech, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Czech, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
