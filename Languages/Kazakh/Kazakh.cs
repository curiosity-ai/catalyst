
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Kazakh
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Kazakh, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Kazakh, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Kazakh, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Kazakh, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Kazakh, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Kazakh, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Kazakh, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Kazakh, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Kazakh, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Kazakh, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Kazakh, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Kazakh).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Kazakh, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Kazakh, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Kazakh, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Kazakh, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Kazakh, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
