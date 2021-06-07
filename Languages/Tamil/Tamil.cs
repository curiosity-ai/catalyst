
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Tamil
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Tamil, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Tamil, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Tamil, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Tamil, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tamil, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Tamil, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tamil, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Tamil, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tamil, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Tamil, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Tamil, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Tamil).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Tamil, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Tamil, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Tamil, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Tamil, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Tamil, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
