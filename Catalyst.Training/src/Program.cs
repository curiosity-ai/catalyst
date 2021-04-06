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
                            if (true) //string.IsNullOrWhiteSpace(options.Token))
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

                            if (!string.IsNullOrWhiteSpace(options.SpacyLookupsData))
                            {
                                await PrepareSpacyLookups.RunAsync(options.SpacyLookupsData, options.LanguagesDirectory);
                            }

                            return;

                            if (!string.IsNullOrWhiteSpace(options.UniversalDependenciesPath))
                            {
                                await TrainSentenceDetector.Train(options.UniversalDependenciesPath, options.LanguagesDirectory);
                                await TrainPOSTagger.Train(udSource: options.UniversalDependenciesPath, ontonotesSource: options.OntonotesPath, languagesDirectory: options.LanguagesDirectory);
                            }

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

                            await CreateProjectsIfNeeded(options.LanguagesDirectory);

                            //if (!string.IsNullOrWhiteSpace(options.FastTextLanguageSentencesPath))
                            //{
                            //    TrainLanguageDetector.Train(options.FastTextLanguageSentencesPath);
                            //    TrainLanguageDetector.Test(options.FastTextLanguageSentencesPath);
                            //}

                            //if (!string.IsNullOrWhiteSpace(options.LanguageJsonPath))
                            //{
                            //    TrainLanguageDetector.CreateLanguageDetector(options.LanguageJsonPath);
                            //}


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
                    var projectFile = $"Catalyst.Models.{lang}.csproj";
                    var projectDir = Path.Combine(dir, projectFile);
                    if (!File.Exists(projectDir))
                    {
                        await File.WriteAllTextAsync(projectDir,
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