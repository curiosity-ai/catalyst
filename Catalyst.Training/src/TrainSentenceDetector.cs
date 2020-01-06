// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Training
{
    public class TrainSentenceDetector
    {
        public static void Train(string udSource)
        {
            var trainFiles = Directory.GetFiles(udSource, "*-train.conllu", SearchOption.AllDirectories);
            var testFiles = Directory.GetFiles(udSource, "*-dev.conllu", SearchOption.AllDirectories);

            var trainFilesPerLanguage = trainFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            var testFilesPerLanguage = testFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            var languages = trainFilesPerLanguage.Keys.ToList();

            Console.WriteLine($"Found these languages for training: {string.Join(", ", languages)}");
            foreach (var forceCase in new EnumCase[] { EnumCase.Original, EnumCase.ForceUpper, EnumCase.ForceLower }) //need tom fix the storage model first - maybe join all in one model
            {
                Parallel.ForEach(languages, new ParallelOptions(), lang =>
                {
                    Language language;
                    try
                    {
                        language = Languages.CodeToEnum(lang);
                    }
                    catch
                    {
                        Console.WriteLine($"Unknown language {lang}");
                        return;
                    }

                    var modelTag = (forceCase == EnumCase.ForceUpper ? "upper" : (forceCase == EnumCase.ForceLower ? "lower" : ""));
                    var sentenceDetector = new SentenceDetector(language, 0, modelTag);

                    var trainDocuments = ReadCorpus(trainFilesPerLanguage[lang], ConvertCase: forceCase,sentenceDetector:sentenceDetector);

                    //TODO: Implement test
                    //if(testFilesPerLanguage.TryGetValue(lang, out var testFile))
                    //{
                    //    var testDocuments = ReadUniversalDependencyCorpus(testFile, ConvertCase: forceCase, sentenceDetector: sentenceDetector);
                    //}

                    Console.WriteLine($"Now training {lang} in mode {forceCase} using files {string.Join(", ", trainFilesPerLanguage[lang])}");
                    sentenceDetector.Train(trainDocuments);
                    sentenceDetector.StoreAsync().Wait();
                });
            }
        }

        private static List<List<SentenceDetector.SentenceDetectorToken>> ReadCorpus(List<string> trainDocuments, EnumCase ConvertCase, SentenceDetector sentenceDetector)
        {
            var allLines = trainDocuments.SelectMany(f => File.ReadAllLines(f));
            var sentences = allLines.Where(l => l.StartsWith("# text =")).Select(l => l.Split(new char[] { '=' }, 2).Last().Trim()).ToList();
            if (ConvertCase == EnumCase.ForceUpper) { sentences = sentences.Select(s => s.ToUpperInvariant()).ToList(); }
            if (ConvertCase == EnumCase.ForceLower) { sentences = sentences.Select(s => s.ToLowerInvariant()).ToList(); }

            return sentences.Select(s =>
            {
                var tk = sentenceDetector.SentenceDetectorTokenizer(s).Select(t => new SentenceDetector.SentenceDetectorToken(t.Value,t.Begin,t.End)).ToList();
                tk.Last().IsSentenceEnd = true;
                return tk;
            }).ToList();
        }

    }

    public enum EnumCase
    {
        Original,
        ForceUpper,
        ForceLower
    }
}