
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Croatian
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Croatian, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Croatian, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Croatian, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Croatian, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Croatian, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Croatian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Croatian, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Croatian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Croatian, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Croatian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Croatian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Croatian).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Croatian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Croatian, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Croatian, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Croatian, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Croatian, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
