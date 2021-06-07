
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Church_Slavic_Slavonic_Old_Bulgarian
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Church_Slavic_Slavonic_Old_Bulgarian).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Church_Slavic_Slavonic_Old_Bulgarian, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Church_Slavic_Slavonic_Old_Bulgarian, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Church_Slavic_Slavonic_Old_Bulgarian, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Church_Slavic_Slavonic_Old_Bulgarian, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Church_Slavic_Slavonic_Old_Bulgarian, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get(), isThreadSafe:true));
        }
    }
}
