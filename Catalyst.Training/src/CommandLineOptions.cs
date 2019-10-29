using CommandLine;
using System;

namespace Catalyst.Training
{
    public class CommandLineOptions
    {
        [Option('s', "storage", Required = true, HelpText = "Path to store data")]
        public string DiskStoragePath { get; set; }

        [Option("ud", HelpText = "Path to the Universal Dependencies folder")]
        public string UniversalDependenciesPath { get; set; }

        [Option("ontonotes", HelpText = "Path to the Ontonotes data folder")]
        public string OntonotesPath { get; set; }

        [Option("langdetect", HelpText = "Path to FastText language detection sentences.csv file")]
        public string FastTextLanguageSentencesPath { get; set; }

        [Option("cld3", HelpText = "Path to cld3 language detection json files")]
        public string LanguageJsonPath { get; set; }

        [Option("wikiner", HelpText = "Path to WikiNER training data")]
        public string WikiNERPath { get; set; }

        [Option('t', "token", HelpText = "Upload token for online repository")]
        public string Token { get; set; }
    }
}
