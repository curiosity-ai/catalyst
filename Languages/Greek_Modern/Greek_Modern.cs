
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Greek_Modern
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Greek_Modern, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Greek_Modern, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Greek_Modern, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Greek_Modern, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Greek_Modern, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Greek_Modern, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Greek_Modern, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Greek_Modern, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Greek_Modern, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Greek_Modern, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Greek_Modern, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Greek_Modern).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Greek_Modern, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Greek_Modern, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Greek_Modern, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Greek_Modern, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Greek_Modern, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
