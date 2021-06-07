
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Hindi
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Hindi, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Hindi, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Hindi, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Hindi, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hindi, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Hindi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hindi, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Hindi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hindi, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Hindi, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Hindi, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Hindi).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Hindi, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Hindi, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Hindi, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Hindi, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Hindi, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
