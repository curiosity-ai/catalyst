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
using Microsoft.Extensions.Logging;

namespace Catalyst.Training
{
    public class TrainSentenceDetector
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<TrainSentenceDetector>();

        public static async Task Train(string udSource, string languagesDirectory)
        {
            var trainFiles = Directory.GetFiles(udSource, "*-train.conllu", SearchOption.AllDirectories);
            var testFiles = Directory.GetFiles(udSource, "*-dev.conllu", SearchOption.AllDirectories);

            var trainFilesPerLanguage = trainFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            var testFilesPerLanguage = testFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            
            var languages = new List<(Language language, string lang)>();

            foreach (var lang in trainFilesPerLanguage.Keys)
            {
                try
                {
                    var language = Languages.CodeToEnum(lang);
                    languages.Add((language, lang));
                }
                catch
                {
                    Logger.LogWarning($"Unknown language {lang}");
                }
            }

            Logger.LogInformation($"Found these languages for training: {string.Join(", ", languages.Select(l => l.language))}");

            foreach (var forceCase in new [] { EnumCase.Original, EnumCase.ForceUpper, EnumCase.ForceLower })
            {
                await Task.WhenAll(languages.Select(async v =>
                {
                    await Task.Yield();

                    var (language, lang) = (v.language, v.lang);

                    var modelTag = (forceCase == EnumCase.ForceUpper ? "upper" : (forceCase == EnumCase.ForceLower ? "lower" : ""));
                    var sentenceDetector = new SentenceDetector(language, 0, modelTag);

                    var trainDocuments = await ReadCorpusAsync(trainFilesPerLanguage[lang], ConvertCase: forceCase, sentenceDetector: sentenceDetector);

                    //TODO: Implement test
                    //if(testFilesPerLanguage.TryGetValue(lang, out var testFile))
                    //{
                    //    var testDocuments = ReadUniversalDependencyCorpus(testFile, ConvertCase: forceCase, sentenceDetector: sentenceDetector);
                    //}

                    Logger.LogInformation($"Now training {lang} in mode {forceCase} using files {string.Join(", ", trainFilesPerLanguage[lang])}");
                    var scoreTest = sentenceDetector.Train(trainDocuments);
                    Logger.LogInformation($"Finished training {lang} in mode {forceCase}");
                    await sentenceDetector.StoreAsync();

                    if (scoreTest > 90)
                    {
                        //Prepare models for new nuget-based distribution
                        var resDir = Path.Combine(languagesDirectory, language.ToString(), "Resources");

                        Directory.CreateDirectory(resDir);

                        using (var f = File.OpenWrite(Path.Combine(resDir, $"sentence-detector{(string.IsNullOrEmpty(modelTag) ? "" : "-" + modelTag)}.bin")))
                        {
                            await sentenceDetector.StoreAsync(f);
                        }
                        await File.WriteAllTextAsync(Path.Combine(resDir, $"sentence-detector{(string.IsNullOrEmpty(modelTag) ? "" : "-" + modelTag)}.score"), $"{scoreTest:0.0}%");

                    }
                }).ToArray());
            }
        }

        private static async Task<List<List<SentenceDetector.SentenceDetectorToken>>> ReadCorpusAsync(List<string> trainDocuments, EnumCase ConvertCase, SentenceDetector sentenceDetector)
        {
            var allLines = new List<string>();
            
            foreach (var file in trainDocuments)
            {
                allLines.AddRange(await File.ReadAllLinesAsync(file));
            }

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