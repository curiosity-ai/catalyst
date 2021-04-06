
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Luxembourgish_Letzeburgesch
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.Luxembourgish_Letzeburgesch, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "tagger.bin",                  async (s) => { var a = new AveragePerceptronTagger(Language.Luxembourgish_Letzeburgesch, 0, "");                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.Luxembourgish_Letzeburgesch, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "parser.bin",                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.Luxembourgish_Letzeburgesch, 0, "");                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "sentence-detector.bin",       async (s) => { var a = new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0, "lower").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "sentence-detector-lower.bin", async (s) => { var a = new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0, "upper").GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "sentence-detector-upper.bin", async (s) => { var a = new SentenceDetector(Language.Luxembourgish_Letzeburgesch, 0, "");                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.Luxembourgish_Letzeburgesch, 0, "WikiNER", new string[] { "Person", "Organization", "Location" }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(Luxembourgish_Letzeburgesch).Assembly, "wikiner.bin",                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.Luxembourgish_Letzeburgesch, 0, "WikiNER", new string[] { "Person", "Organization", "Location" });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.Luxembourgish_Letzeburgesch, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.Luxembourgish_Letzeburgesch, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.Luxembourgish_Letzeburgesch, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.Luxembourgish_Letzeburgesch, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
