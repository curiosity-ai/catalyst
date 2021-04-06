using Mosaik.Core;
using System;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging.Console;
using System.Threading.Tasks;
using CommandLine;
using Microsoft.Extensions.Logging;
using System.Text;
using System.IO;

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
                            if (true || string.IsNullOrWhiteSpace(options.Token))
                            {
                                Storage.Current = new DiskStorage(options.DiskStoragePath);
                            }
                            else
                            {
                                //For uploading on the online models repository
                                Storage.Current = new OnlineWriteableRepositoryStorage(new DiskStorage(options.DiskStoragePath), options.Token);
                            }

                            Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
                            ThreadPool.SetMinThreads(Environment.ProcessorCount*2, Environment.ProcessorCount*2);
                            ThreadPool.SetMaxThreads(Environment.ProcessorCount*20, Environment.ProcessorCount*20);

                            using (var p = Process.GetCurrentProcess())
                            {
                                p.PriorityClass = ProcessPriorityClass.High;
                            }

                            await CreateProjectsIfNeeded(options.LanguagesDirectory);

                            if (!string.IsNullOrWhiteSpace(options.SpacyLookupsData))
                            {
                                await PrepareSpacyLookups.RunAsync(options.SpacyLookupsData, options.LanguagesDirectory);
                            }

                            if (!string.IsNullOrWhiteSpace(options.UniversalDependenciesPath))
                            {
                                //await TrainSentenceDetector.Train(options.UniversalDependenciesPath, options.LanguagesDirectory);
                                await TrainPOSTagger.Train(udSource: options.UniversalDependenciesPath, ontonotesSource: options.OntonotesPath, languagesDirectory: options.LanguagesDirectory);
                            }
                            return;

                            if (!string.IsNullOrWhiteSpace(options.WikiNERPath))
                            {
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.English, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.French, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.German, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Spanish, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Italian, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Portuguese, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Russian, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Dutch, 0, "WikiNER", options.LanguagesDirectory);
                                await TrainWikiNER.TrainAsync(options.WikiNERPath, Language.Polish, 0, "WikiNER", options.LanguagesDirectory);
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

        private static async Task CreateProjectsIfNeeded(string languagesDirectory)
        {
            foreach(var dir in Directory.GetDirectories(languagesDirectory))
            {
                var dirInfo = new DirectoryInfo(dir);
                if(Enum.TryParse<Language>(dirInfo.Name, out var lang))
                {
                    var projectFile    = Path.Combine(dir, $"Catalyst.Models.{lang}.csproj");
                    var langFile       = Path.Combine(dir, $"{lang}.cs");
                    var lemmatizerFile = Path.Combine(dir, $"{lang}.Lemmatizer.cs");
                    var stopwordsFile  = Path.Combine(dir, $"{lang}.StopWords.cs");
                    var exceptionsFile = Path.Combine(dir, $"{lang}.TokenizerExceptions.cs");

                    if (!File.Exists(projectFile))
                    {
                        await File.WriteAllTextAsync(projectFile,
@"
<Project Sdk='Microsoft.NET.Sdk'>
  <PropertyGroup>
    <TargetFramework>netstandard2.1</TargetFramework>
    <Platforms>AnyCPU</Platforms>
    <LangVersion>latest</LangVersion>
  </PropertyGroup>

  <PropertyGroup>
    <Authors>Curiosity GmbH</Authors>
    <Copyright>(c) Copyright 2021 Curiosity GmbH - all right reserved</Copyright>
    <PackageLicenseExpression>MIT</PackageLicenseExpression>
    <PackageProjectUrl>www.curiosity.ai</PackageProjectUrl>
    <RepositoryUrl>https://github.com/curiosity-ai/catalyst</RepositoryUrl>
    <GeneratePackageOnBuild>true</GeneratePackageOnBuild>
    <PackageRequireLicenseAcceptance>false</PackageRequireLicenseAcceptance>
    <PackageId>Catalyst.Models.English</PackageId>
    <Description>This package contains the default English models for Catalyst. Catalyst is a Natural Language Processing library built from scratch for speed. Inspired by spaCy's design, it brings pre-trained models, out-of-the box support for training word and document embeddings, and flexible entity recognition models.</Description>
    <PackageTags>English, Natural Language Processing, NLP, Spacy, Machine Learning, ML, Embeddings, Data Science, Big Data, Artificial Intelligence, AI, NLP Library, Neural Networks, Deep Learning</PackageTags>
    <PackageIcon>catalyst-logo.png</PackageIcon>
  </PropertyGroup>


  <ItemGroup>
    <None Remove='Resources\*.bin' />
    <EmbeddedResource Include='Resources\*.bin' />
  </ItemGroup>

  <ItemGroup>
    <None Include='../../Catalyst/catalyst-logo.png'>
      <Pack>True</Pack>
      <PackagePath></PackagePath>
    </None>
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include='Mosaik.Core' Version='0.0.16583' />
    <PackageReference Include='Catalyst' Version='0.0.16396' />
  </ItemGroup>
</Project>
".Replace("'", "\"").Replace("English", lang.ToString()));
                    }

                    if(!File.Exists(langFile))
                    {
                        await File.WriteAllTextAsync(langFile,
@"
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.English, 0).GetStoredObjectInfo(),                                                                             async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'tagger.bin',                  async (s) => { var a = new AveragePerceptronTagger(Language.English, 0, '');                                                                          await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronDependencyParser(Language.English, 0).GetStoredObjectInfo(),                                                                   async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'parser.bin',                  async (s) => { var a = new AveragePerceptronDependencyParser(Language.English, 0, '');                                                                await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0).GetStoredObjectInfo(),                                                                                    async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'sentence-detector.bin',       async (s) => { var a = new SentenceDetector(Language.English, 0, '');                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0, 'lower').GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'sentence-detector-lower.bin', async (s) => { var a = new SentenceDetector(Language.English, 0, '');                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new SentenceDetector(Language.English, 0, 'upper').GetStoredObjectInfo(),                                                                           async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'sentence-detector-upper.bin', async (s) => { var a = new SentenceDetector(Language.English, 0, '');                                                                                 await a.LoadAsync(s); return a; }));
            ObjectStore.OverrideModel(new AveragePerceptronEntityRecognizer(Language.English, 0, 'WikiNER', new string[] { 'Person', 'Organization', 'Location' }).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, 'wikiner.bin',                 async (s) => { var a = new AveragePerceptronEntityRecognizer(Language.English, 0, 'WikiNER', new string[] { 'Person', 'Organization', 'Location' });  await a.LoadAsync(s); return a; }));

            Catalyst.StopWords.Snowball.Register(Language.English, StopWords.Snowball);
            Catalyst.StopWords.Spacy.Register(Language.English, StopWords.Spacy);
            Catalyst.LemmatizerStore.Register(Language.English, new Lemmatizer());
            Catalyst.TokenizerExceptions.Register(Language.English, new Lazy<Dictionary<int, TokenizationException>>(() => TokenizerExceptions.Get()));
        }
    }
}
".Replace("'", "\"").Replace("English", lang.ToString()));
                    }

                    if (!File.Exists(lemmatizerFile))
                    {
                        await File.WriteAllTextAsync(lemmatizerFile,
@"
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.English;

            public string GetLemma(IToken token)
            {
                return token.Value;
            }

            public ReadOnlySpan<char> GetLemmaAsSpan(IToken token)
            {
                return token.ValueAsSpan;
            }

            public bool IsBaseForm(IToken token)
            {
                return false;
            }
        }
    }
}
".Replace("'", "\"").Replace("English", lang.ToString()));
                    }

                    if (true) //!File.Exists(stopwordsFile))
                    {
                        await File.WriteAllTextAsync(stopwordsFile,
@"
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static class StopWords
        {
            public static ReadOnlyHashSet<string> Snowball = new ReadOnlyHashSet<string>(new HashSet<string>(new string[] { }));
            public static ReadOnlyHashSet<string> Spacy    = new ReadOnlyHashSet<string>(new HashSet<string>(new string[] { }));
        }
    }
}
".Replace("'", "\"").Replace("English", lang.ToString()));
                    }

                    if (!File.Exists(exceptionsFile))
                    {
                        await File.WriteAllTextAsync(exceptionsFile,
@"
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        internal sealed class TokenizerExceptions 
        {
            internal static Dictionary<int, TokenizationException> Get()
            {
                var exceptions = Catalyst.TokenizerExceptions.CreateBaseExceptions();
                return exceptions;
            }
        }
    }
}
".Replace("'", "\"").Replace("English", lang.ToString()));
                    }
                }
            }
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