
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Norwegian_Bokmal
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Norwegian_Bokmal, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Norwegian_Bokmal, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Norwegian_Bokmal, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Norwegian_Bokmal, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Norwegian_Bokmal, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Norwegian_Bokmal, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Norwegian_Bokmal, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Norwegian_Bokmal, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Norwegian_Bokmal, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Norwegian_Bokmal, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Norwegian_Bokmal, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Norwegian_Bokmal).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Norwegian_Bokmal, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Norwegian_Bokmal, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Norwegian_Bokmal, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Norwegian_Bokmal, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Norwegian_Bokmal, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
