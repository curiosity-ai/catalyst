
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Afrikaans
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Afrikaans, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Afrikaans, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Afrikaans, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Afrikaans, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Afrikaans, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Afrikaans, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Afrikaans, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Afrikaans, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Afrikaans, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Afrikaans, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Afrikaans, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Afrikaans).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Afrikaans, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Afrikaans, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Afrikaans, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Afrikaans, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Afrikaans, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), System.Threading.LazyThreadSafetyMode.ExecutionAndPublication));
        }
    }
}
