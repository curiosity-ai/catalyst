using Mosaik.Core;
using System;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging.Console;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.Extensions.Logging;
using System.Text;

namespace Catalyst.Training
{
    class Program
    {
        static async Task Main(string[] args)
        {
            ApplicationLogging.SetLoggerFactory(LoggerFactory.Create(builder => builder.AddConsole()));
            ForceInvariantCultureAndUTF8Output();

            await Parser.Default
                        .ParseArguments<CommandLineOptions>(args)
                        .MapResult(
                        async options =>
                        {
                            if (string.IsNullOrWhiteSpace(options.Token))
                            {
                                Storage.Current = new DiskStorage(options.DiskStoragePath);
                            }
                            else
                            {
                                //For uploading on the online models repository
                                Storage.Current = new OnlineWriteableRepositoryStorage(new DiskStorage(options.DiskStoragePath), options.Token);
                            }

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
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.English, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.French, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.German, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Spanish, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Italian, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Portuguese, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Russian, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Dutch, 0, "WikiNER");
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Polish, 0, "WikiNER");
                            }

                            if (!string.IsNullOrWhiteSpace(options.FastTextLanguageSentencesPath))
                            {
                                TrainLanguageDetector.Train(options.FastTextLanguageSentencesPath);
                                TrainLanguageDetector.Test(options.FastTextLanguageSentencesPath);
                            }

                            if (!string.IsNullOrWhiteSpace(options.LanguageJsonPath))
                            {
                                TrainLanguageDetector.CreateLanguageDetector(options.LanguageJsonPath);
                            }

                        },
                        error => Task.CompletedTask);
        }

        private static void ForceInvariantCultureAndUTF8Output()
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;
            Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
        }

    }
}