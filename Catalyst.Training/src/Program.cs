using Mosaik.Core;
using System;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging.Console;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.Extensions.Logging;

namespace Catalyst.Training
{
    class Program
    {
        static void Main(string[] args)
        {
            ApplicationLogging.SetLoggerFactory(new LoggerFactory().AddConsole());
            Console.InputEncoding = System.Text.Encoding.Unicode;
            Console.OutputEncoding = System.Text.Encoding.Unicode;

            Parser.Default
                        .ParseArguments<CommandLineOptions>(args)
                        .MapResult(
                        options =>
                        {
                            Storage.Current = new DiskStorage(options.DiskStoragePath);
                            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
                            using (var p = Process.GetCurrentProcess())
                            {
                                p.PriorityClass = ProcessPriorityClass.High;
                            }

                            if (!string.IsNullOrWhiteSpace(options.UniversalDependenciesPath))
                            {
                                TrainSentenceDetector.Train(options.UniversalDependenciesPath);
                                TrainPOSTagger.Train(udSource: options.UniversalDependenciesPath, ontonotesSource: options.OntonotesPath);
                            }

                            if (!string.IsNullOrWhiteSpace(options.WikiNERPath))
                            {
                                TrainWikiNER.Train(options.WikiNERPath, Language.English,    0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.French,     0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.German,     0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Spanish,    0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Italian,    0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Portuguese, 0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Russian,    0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Dutch,      0, "WikiNER");
                                TrainWikiNER.Train(options.WikiNERPath, Language.Polish,     0, "WikiNER");
                            }

                            if (!string.IsNullOrWhiteSpace(options.FastTextLanguageSentencesPath))
                            {
                                TrainLanguageDetector.Train(options.FastTextLanguageSentencesPath);
                                TrainLanguageDetector.Test(options.FastTextLanguageSentencesPath);
                            }
                            return true;
                        },
                        error => false);
        }
    }
}