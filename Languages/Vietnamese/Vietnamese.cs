
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Vietnamese
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Vietnamese, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Vietnamese, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Vietnamese, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Vietnamese, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Vietnamese, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Vietnamese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Vietnamese, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Vietnamese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Vietnamese, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Vietnamese, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Vietnamese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Vietnamese).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Vietnamese, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Vietnamese, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Vietnamese, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Vietnamese, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Vietnamese, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
