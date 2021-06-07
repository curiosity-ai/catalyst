
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Wolof
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Wolof, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Wolof, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Wolof, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Wolof, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Wolof, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Wolof, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Wolof, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Wolof, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Wolof, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Wolof, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Wolof, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Wolof).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Wolof, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Wolof, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Wolof, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Wolof, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Wolof, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
