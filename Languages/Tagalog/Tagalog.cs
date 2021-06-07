
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Tagalog
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Tagalog, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Tagalog, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Tagalog, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Tagalog, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tagalog, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Tagalog, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tagalog, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Tagalog, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Tagalog, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Tagalog, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Tagalog, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Tagalog).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Tagalog, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Tagalog, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Tagalog, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Tagalog, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Tagalog, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
