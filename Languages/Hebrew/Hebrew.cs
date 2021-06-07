
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Hebrew
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Hebrew, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Hebrew, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Hebrew, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Hebrew, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hebrew, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Hebrew, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hebrew, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Hebrew, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Hebrew, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Hebrew, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Hebrew, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Hebrew).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Hebrew, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Hebrew, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Hebrew, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Hebrew, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Hebrew, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
