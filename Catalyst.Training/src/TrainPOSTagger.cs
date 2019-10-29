// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Catalyst.Training
{
    public class TrainPOSTagger
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<TrainPOSTagger>();

        public static void Train(string udSource, string ontonotesSource)
        {
            var trainFiles = Directory.GetFiles(udSource, "*-train.conllu", SearchOption.AllDirectories);
            var testFiles  = Directory.GetFiles(udSource, "*-dev.conllu", SearchOption.AllDirectories);

            List<string> trainFilesOntonotesEnglish = null;

            if (!string.IsNullOrWhiteSpace(ontonotesSource))
            {
                trainFilesOntonotesEnglish = Directory.GetFiles(ontonotesSource, "*.parse.ddg", SearchOption.AllDirectories)
                                                      .Where(fn => !fn.Contains("sel_") || int.Parse(Path.GetFileNameWithoutExtension(fn).Split(new char[] { '_', '.' }).Skip(1).First()) < 3654)
                                                      .ToList();
            }

            var trainFilesPerLanguage = trainFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            var testFilesPerLanguage = testFiles.Select(f => new { lang = Path.GetFileNameWithoutExtension(f).Replace("_", "-").Split(new char[] { '-' }).First(), file = f }).GroupBy(f => f.lang).ToDictionary(g => g.Key, g => g.Select(f => f.file).ToList());
            var languages = trainFilesPerLanguage.Keys.ToList();

            Logger.LogInformation($"Found these languages for training: {string.Join(", ", languages)}");

            int N_training = 5;

            Parallel.ForEach(languages, lang =>
            {
                Language language;
                try
                {
                    language = Languages.CodeToEnum(lang);
                }
                catch
                {
                    Logger.LogWarning($"Unknown language {lang}");
                    return;
                }

                var arcNames = new HashSet<string>();

                if (trainFilesPerLanguage.TryGetValue(lang, out var langTrainFiles) && testFilesPerLanguage.TryGetValue(lang, out var langTestFiles))
                {
                    var trainDocuments = ReadCorpus(langTrainFiles, arcNames, language);
                    var testDocuments  = ReadCorpus(langTestFiles, arcNames, language);

                    if (language == Language.English)
                    {
                        //Merge with Ontonotes 5.0 corpus
                        trainDocuments.AddRange(ReadCorpus(trainFilesOntonotesEnglish, arcNames, language, isOntoNotes: true));
                    }

                    double bestScore = double.MinValue;

                    for (int i = 0; i < N_training; i++)
                    {
                        var Tagger = new AveragePerceptronTagger(language, 0);
                        Tagger.Train(trainDocuments.AsEnumerable(), (int)(5 + ThreadSafeRandom.Next(15)));
                        var scoreTrain = TestTagger(trainDocuments, Tagger);
                        var scoreTest = TestTagger(testDocuments, Tagger);
                        if (scoreTest > bestScore)
                        {
                            Logger.LogInformation($"\n>>>>> {lang}: NEW POS BEST: {scoreTest:0.0}%");
                            try
                            {
                                Tagger.StoreAsync().Wait();
                            }
                            catch (Exception E)
                            {
                                Logger.LogError(E, $"\n>>>>> {lang}: Failed to store model");
                            } 
                            bestScore = scoreTest;
                        }
                        else
                        {
                            Logger.LogInformation($"\n>>>>> {lang}: POS BEST IS STILL : {bestScore:0.0}%");
                        }
                    }


                    bestScore = double.MinValue;
                    for (int i = 0; i < N_training; i++)
                    {
                        var Parser = new AveragePerceptronDependencyParser(language, 0/*, arcNames.ToList()*/);
                        try
                        {
                            Parser.Train(trainDocuments.AsEnumerable(), (int)(5 + ThreadSafeRandom.Next(10)), (float)(1D - ThreadSafeRandom.NextDouble() * ThreadSafeRandom.NextDouble()));
                        }
                        catch (Exception E)
                        {
                            Logger.LogInformation("FAIL: " + E.Message);
                            continue;
                        }

                        trainDocuments = ReadCorpus(langTrainFiles, arcNames, language);
                        testDocuments  = ReadCorpus(langTestFiles,  arcNames, language);

                        if (language == Language.English)
                        {
                            //Merge with Ontonotes 5.0 corpus
                            trainDocuments.AddRange(ReadCorpus(trainFilesOntonotesEnglish, arcNames, language, isOntoNotes: true));
                        }

                        var scoreTrain = TestParser(trainDocuments, Parser);
                        var scoreTest = TestParser(testDocuments, Parser);

                        if (scoreTest > bestScore)
                        {
                            Logger.LogInformation($"\n>>>>> {lang}: NEW DEP BEST: {scoreTest:0.0}%");
                            try
                            {
                                Parser.StoreAsync().Wait();
                            }
                            catch (Exception E)
                            {
                                Logger.LogError(E, $"\n>>>>> {lang}: Failed to store model");
                            }
                            bestScore = scoreTest;
                        }
                        else
                        {
                            Logger.LogInformation($"\n>>>>> {lang}: DEP BEST IS STILL : {bestScore:0.0}%");
                        }
                        Parser = null;
                    }
                }
            });

            foreach (var lang in languages)
            {
                Language language;
                try
                {
                    language = Languages.CodeToEnum(lang);
                }
                catch
                {
                    Logger.LogInformation($"Unknown language {lang}");
                    return;
                }

                var arcNames = new HashSet<string>();

                var trainDocuments = ReadCorpus(trainFilesPerLanguage[lang], arcNames, language);
                var testDocuments  = ReadCorpus(testFilesPerLanguage[lang],  arcNames, language);

                if (language == Language.English)
                {
                    //Merge with Ontonotes 5.0 corpus
                    var ontonotesDocuments = ReadCorpus(trainFilesOntonotesEnglish,  arcNames, language, isOntoNotes: true);
                    trainDocuments.AddRange(ontonotesDocuments);
                }

                var Tagger = AveragePerceptronTagger.FromStoreAsync(language, 0, "").WaitResult();
                Logger.LogInformation($"\n{lang} - TAGGER / TRAIN");
                TestTagger(trainDocuments, Tagger);

                Logger.LogInformation($"\n{lang} - TAGGER / TEST");
                TestTagger(testDocuments, Tagger);

                trainDocuments = ReadCorpus(trainFilesPerLanguage[lang], arcNames, language);
                testDocuments  = ReadCorpus(testFilesPerLanguage[lang],  arcNames, language);

                var Parser = AveragePerceptronDependencyParser.FromStoreAsync(language, 0, "").WaitResult();
                Logger.LogInformation($"\n{lang} - PARSER / TRAIN");
                TestParser(trainDocuments, Parser);

                Logger.LogInformation($"\n{lang} - PARSER / TEST");
                TestParser(testDocuments, Parser);
            }
        }

        private static double TestParser(List<IDocument> testDocuments, AveragePerceptronDependencyParser parser)
        {
            var sentences = testDocuments.SelectMany(d => d.Spans).Where(s => s.IsProjective() && s.TokensCount > 4 && !s.Any(tk => tk.Value.Contains("@") || tk.Value.Contains("://"))).ToList();
            int correctUnlabeled = 0, correctLabeled = 0, total = 0, correctRoot = 0;
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            Parallel.ForEach(sentences, s =>
            {
                var goldHeads  = s.Select(tk => tk.Head).ToArray();
                var goldLabels = s.Select(tk => tk.DependencyType).ToArray();
                parser.Predict(s);
                int UAS = 0, LAS = 0, tot = 0,ROOT = 0;
                for(int i = 0; i < s.TokensCount; i++)
                {
                    //if (!s[i].Value.Any(c=> char.IsPunctuation(c)))
                    {
                        if(goldHeads[i] == -1)
                        {
                            ROOT += (goldHeads[i] == s[i].Head) ? 1 : 0;
                        }
                        bool correctHead  =  goldHeads[i] == s[i].Head;
                        bool correctLabel = goldLabels[i] == s[i].DependencyType;

                        if (correctHead) { UAS++; }
                        if (correctHead && correctLabel) { LAS++; }
                        tot++;
                    }
                    //Restore original values
                    s[i].Head = goldHeads[i];
                    s[i].DependencyType = goldLabels[i];
                }
                Interlocked.Add(ref correctUnlabeled, UAS);
                Interlocked.Add(ref correctLabeled, LAS);
                Interlocked.Add(ref total, tot);
                Interlocked.Add(ref correctRoot, ROOT);
            });
            sw.Stop();
            double UASscore = 100D * correctUnlabeled / total;
            Logger.LogInformation($"UAS:{UASscore:0.00}% & LAS:{100D * correctLabeled / total:0.00}% & & R:{100D * correctRoot / sentences.Count:0.00}% @ {1000D * total / sw.ElapsedMilliseconds:0} tokens/second");

            return UASscore;
        }

        private static object lockMistake = new object();

        private static double TestTagger(List<IDocument> testDocuments, AveragePerceptronTagger Tagger)
        {
            var sentences = testDocuments.SelectMany(d => d.Spans).ToList();
            int correct = 0, total = 0;
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            int TP = 0, FN = 0, FP = 0;

            Parallel.ForEach(sentences, s =>
            {
                var tags = s.Tokens.Select(t => t.POS).ToArray();
                Tagger.Predict(s);
                var pred = s.Tokens.Select(t => t.POS).ToArray();
                int correctOnSentence = tags.Zip(pred, (t, p) => t == p ? 1 : 0).Sum();

                int _TP = 0, _FN = 0, _FP = 0;

                for (int m = 0; m < tags.Length; m++)
                {
                    if (tags[m] == pred[m]) { TP++; }
                    if (tags[m] != pred[m]) { FP++; FN++; } //Same if we are not evaluating per-tag precision / recall
                }

                Interlocked.Add(ref TP, _TP);
                Interlocked.Add(ref FN, _FN);
                Interlocked.Add(ref FP, _FP);

                if (correctOnSentence < s.TokensCount)
                {
                    var sb = new StringBuilder();

                    for (int m = 0; m < tags.Length; m++)
                    {
                        sb.Append(s[m].Value);
                        if(tags[m] != pred[m])
                        {
                            sb.Append("[").Append("P:").Append(pred[m]).Append(" C:").Append(tags[m]).Append("]");
                        }
                        sb.Append(" ");
                    }
                    sb.AppendLine();

                    lock (lockMistake)
                    {
                        File.AppendAllText("mistakes.txt", sb.ToString());
                    }
                }

                Interlocked.Add(ref correct,correctOnSentence);
                Interlocked.Add(ref total, s.TokensCount);

                for(int i = 0; i < s.TokensCount; i++)
                {
                    s[i].POS = tags[i];
                }

            });
            sw.Stop();

            Logger.LogInformation($"POS: {Math.Round(100D * correct / total, 2)}% at a rate of {Math.Round(1000D * total / sw.ElapsedMilliseconds, 0) } tokens/second");

            var precision = (double)TP / (TP + FP);
            var recall = (double)TP / (TP + FN);

            Logger.LogInformation($"F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% ");

            return 100D * correct / total;
        }

        private static List<IDocument> ReadCorpus(List<string> trainDocuments, HashSet<string> arcNames, Language language, bool isOntoNotes = false)
        {
            if(trainDocuments is null)
            {
                return new List<IDocument>();
            }
            var allLines = new List<string>();

            foreach (var f in trainDocuments)
            {
                if(isOntoNotes)
                {
                    allLines.Add("# newdoc"); //Force doc splits
                    allLines.Add("# sent_id"); //Force doc splits
                    allLines.AddRange(File.ReadAllLines(f).Select(l => string.IsNullOrWhiteSpace(l) ? "# sent_id" : l));
                }
                else
                {
                    allLines.AddRange(File.ReadAllLines(f).Where(l => !string.IsNullOrWhiteSpace(l)));
                }
            }

            var documents = new List<IDocument>();


            var docLines = new List<List<string>>();

            foreach(var line in allLines)
            {
                if(line.StartsWith("# newdoc"))
                {
                    docLines.Add(new List<string>());
                }
                else
                {
                    if (docLines.Count == 0) { docLines.Add(new List<string>()); }
                    docLines.Last().Add(line);
                }
            }

            foreach(var docline in docLines)
            {
                var doc = new Document();
                bool invalidDoc = false;

                ISpan span = null;
                var sb = new StringBuilder();
                foreach (var l in docline)
                {
                    if (l.StartsWith("# sent_id"))
                    {
                        span = doc.AddSpan(sb.Length, sb.Length);
                        //if(l.Contains("email-enronsent")) { invalidDoc = true; }
                    }
                    else if (!l.StartsWith("#"))
                    {
                        var parts = l.Split('\t');
                        if (parts[0].Contains("-")) { continue; } //Pseudo-token, such as cannot -> proceed by can + not

                        double index;
                        if (double.TryParse(parts[0], out index))
                        {
                            if((int)(index*10) == ((int)index) * 10)
                            {
                                string txt   = parts[1];
                                string lemma = parts[2];
                                string pos   = parts[3];
                                PartOfSpeech POS;
                                bool spaceAfter = false;


                                if (isOntoNotes)
                                {
                                    POS = PartOfSpeechHelpers.EnglishPennToUniversal[pos];
                                }
                                else
                                {
                                    POS = (PartOfSpeech)Enum.Parse(typeof(PartOfSpeech), pos);
                                    spaceAfter = parts[9].Contains("SpaceAfter=No");
                                }
                                //if (PartOfSpeechHelpers.StringPOS.Contains(pos))
                                //{
                                //    POS = (PartOfSpeechEnum)Enum.Parse(typeof(PartOfSpeechEnum), pos);
                                //}
                                //else
                                //{
                                //    if (language == LanguageEnum.English)
                                //    {
                                //        if (!PartOfSpeechHelpers.EnglishPennToUniversal.TryGetValue(pos, out POS))
                                //        {
                                //            throw new Exception("Invalid tag: " + pos);
                                //        }
                                //    }
                                //    else
                                //    {
                                //        throw new Exception("Invalid tag: " + pos);
                                //    }
                                //}

                                if(language  == Language.English)
                                {
                                    //Should add more exceptions here on how we handle tokenization differently than the original Conll data
                                    if ((txt.ToLowerInvariant() == "'s" || txt.ToLowerInvariant() == "s") && (lemma.ToLowerInvariant() == "be" || POS == PartOfSpeech.VERB || POS == PartOfSpeech.AUX))
                                    {
                                        txt = "is";
                                    }
                                    else if ((txt.ToLowerInvariant() == "'m" || txt.ToLowerInvariant() == "m") && (lemma.ToLowerInvariant() == "be" || POS == PartOfSpeech.VERB || POS == PartOfSpeech.AUX))
                                    {
                                        txt = "am";
                                    }
                                    else if ((txt.ToLowerInvariant() == "'re" || txt.ToLowerInvariant() == "re" )&& (lemma.ToLowerInvariant() == "be" || POS == PartOfSpeech.VERB))
                                    {
                                        txt = "are";
                                    }
                                    else if ((txt.ToLowerInvariant() == "ll" || txt.ToLowerInvariant() == "'ll") && (POS == PartOfSpeech.VERB || POS == PartOfSpeech.AUX))
                                    {
                                        txt = "will";
                                    }
                                    else if (txt.ToLowerInvariant() == "'d" && (POS == PartOfSpeech.AUX))
                                    {
                                        txt = "would";
                                    }
                                    else if (txt.ToLowerInvariant() == "'d" && (POS == PartOfSpeech.VERB))
                                    {
                                        txt = "had";
                                    }
                                    else if (txt.ToLowerInvariant() == "n't")
                                    {
                                        txt = "not";
                                    }
                                    else if (txt.ToLowerInvariant() == "'ve")
                                    {
                                        txt = "have";
                                    }
                                    else if (txt.Length > 1 && txt.StartsWith("/") && pos == ".")
                                    {
                                        txt = txt.Substring(1);
                                    }
                                    else if(txt =="'" && lemma == "'s" && (POS == PartOfSpeech.PART || POS == PartOfSpeech.PRON))
                                    {
                                        // ok
                                    }
                                    else if(txt.StartsWith("'") && !(txt == "'s" && POS == PartOfSpeech.PART)
                                                                && !(txt == "'"  && POS == PartOfSpeech.PART)
                                                                && !(txt == "'s" && POS == PartOfSpeech.PRON)
                                                                && !(txt == "'"  && POS == PartOfSpeech.PUNCT) )
                                    {
                                        File.AppendAllLines("missing_contractions.txt", new string[] { l.Split(new char[] { '\t' }, 2).Last() });
                                    }
                                    else if(lemma == "#hlink#" && txt.Contains("://"))
                                    {
                                        txt = "http://" + txt;
                                    }
                                }
                                //'d

                                int begin = sb.Length;
                                int end = begin + txt.Length - 1;
                                sb.Append(txt + " ");
                                span.End = sb.Length - 1;
                                var token = span.AddToken(begin, end);
                                token.POS = POS;
                                int head = int.Parse(parts[isOntoNotes ? 5 : 6])-1;
                                string arcType = parts[isOntoNotes ? 6 : 7].ToLowerInvariant().Split(':').First();

                                //if (parts[5].Contains("Foreign=Yes"))
                                //{
                                //    invalidDoc = true;
                                //}

                                token.Head = head;
                                token.DependencyType = arcType;

                                if (!arcNames.Contains(arcType)) { arcNames.Add(arcType); }
                            }
                        }
                    }
                }
                doc.Value = sb.ToString();
                doc.TrimTokens();
                if (!invalidDoc)
                {
                    documents.Add(doc);
                }
                else
                {
                    Logger.LogInformation("skipping document:\n" + doc.TokenizedValue + "\n");
                }
            }


            return documents;
        }

    }
}