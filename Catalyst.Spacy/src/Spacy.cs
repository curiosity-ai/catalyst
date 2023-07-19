﻿using Mosaik.Core;
using Python.Deployment;
using Python.Runtime;
using Semver;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace Catalyst
{
    public static partial class Spacy
    {
        private const string DownloadURL = "https://github.com/explosion/spacy-models/releases/download";
        private const string CompatibilityDataURL = "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json";
        private const string ShortcutsURL = "https://raw.githubusercontent.com/explosion/spacy-models/master/shortcuts-v2.json";

        public static Dictionary<string, Dictionary<string, Dictionary<string, string[]>>> CompatibilityData { get; private set; }
        public static Dictionary<string, string> Shortcuts { get; private set; }
        public static string SpacyVersion { get; private set; }

        private static Dictionary<string, Pipeline> _pipelines = new Dictionary<string, Pipeline>();
        private static IntPtr _threadState;
        private static bool _disposed = false;
        private static readonly object _lock = new object();

        public static async Task<PythonLock> Initialize(ModelSize modelSize, params Language[] languages)
        {
            await Installer.SetupPython();
            Installer.TryInstallPip();
            Installer.PipInstallModule("spacy");
            Installer.PipInstallModule("spacy-lookups-data");
            PythonEngine.Initialize();

            _threadState = PythonEngine.BeginAllowThreads();

            TestPythonVersion();

            await LoadModelsDataAsync();

            LoadModelsData(modelSize, languages);

            return new PythonLock();
        }

        public static Pipeline For(ModelSize modelSize, Language language)
        {
            using (Py.GIL())
            {
                var key = $"{language}-{modelSize}";

                if (_pipelines.TryGetValue(key, out var pipeline)) return pipeline;

                dynamic spacy = Py.Import("spacy");
                var modelName = GetModelName(modelSize, language);
                var nlp = spacy.load(modelName);
                pipeline = new Pipeline(language, modelSize, nlp);

                _pipelines[key] = pipeline;

                return pipeline;
            }
        }

        private static async Task LoadModelsDataAsync()
        {
            if (CompatibilityData is null || Shortcuts is null)
            {
                using (var client = new HttpClient())
                {
                    CompatibilityData = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, Dictionary<string, string[]>>>>(await client.GetStringAsync(CompatibilityDataURL));
                    Shortcuts = JsonSerializer.Deserialize<Dictionary<string, string>>(await client.GetStringAsync(ShortcutsURL));
                }
            }
        }

        private static void LoadModelsData(ModelSize modelSize, params Language[] languages)
        {
            if (SpacyVersion is null)
            {
                SpacyVersion = GetSpacyVersion();
            }

            foreach (var lang in languages)
            {
                string modelName = GetModelName(modelSize, lang);

                var modelVersion = CompatibilityData["spacy"][SpacyVersion][modelName][0];

                var url = $"{DownloadURL}/{modelName}-{modelVersion}/{modelName}-{modelVersion}.tar.gz#egg={modelName}=={modelVersion}";
                Console.WriteLine(url);
                Installer.PipInstallModule(url);
            }
        }

        private static string GetModelName(ModelSize modelSize, Language lang)
        {
            var modelName = Shortcuts[lang == Language.Any ? "xx" : Languages.EnumToCode(lang)].Replace("_sm", "");
            var size = modelSize == ModelSize.Small ? "sm" : modelSize == ModelSize.Medium ? "md" : "lg";
            modelName += $"_{size}";
            return modelName;
        }

        private static string GetSpacyVersion()
        {
            string version = null;

            using (Py.GIL())
            {
                dynamic spacy = Py.Import("spacy");

                version = spacy.__version__;
            }
            SemVersion currentSpacyVersion = SemVersion.TryParse(version, SemVersionStyles.Any, out var a) ? a : null;
            if (currentSpacyVersion == null)
                throw new Exception($"Unable to parse the current spacy module version ({version}) into a proper semantic version.");

            // Attempt to find an exact match for the current version of spacy in the CompatibilityData
            Console.WriteLine($"Current version of the spacy module: {currentSpacyVersion}");
            Console.WriteLine($"Seaching compatibility data for version {currentSpacyVersion}");
            if (!CompatibilityData["spacy"].ContainsKey(currentSpacyVersion.ToString()))
            {
                Console.WriteLine("Exact match not found.  Searching for best possible match...");

                // Look for the most recent version that isn't greater than the current version
                var fixedVersion = CompatibilityData["spacy"].Keys
                                     .Select((k) => (key: k, semVer: SemVersion.TryParse(k, SemVersionStyles.Any, out var keyVersion) ? keyVersion : null))
                                     .Where(k =>
                                        k.semVer != null // Must be valid semantic version
                                        && SemVersion.ComparePrecedence(currentSpacyVersion, k.semVer) >= 0  // Can't grab a newer release
                                        && k.semVer.IsRelease)  // No prereleases 
                                        // Could also force same major version by adding '&& currentSpacyVersion.Major == k.semVer.Major'
                                        // if backward compatibility across major releases is an issue
                                     .OrderByDescending(k => k.semVer)
                                     .Select(k => k.key)
                                     .FirstOrDefault();

                if (fixedVersion is null)
                {
                    throw new Exception($"Failed to read spacy version from compatibility data. {Environment.NewLine}Expected: {version}{Environment.NewLine}available: {string.Join(", ", CompatibilityData["spacy"].Keys)}");
                }
                version = fixedVersion;
            }

            Console.WriteLine("Spacy version: " + version);
            return version;
        }

        private static void TestPythonVersion()
        {
            using (Py.GIL())
            {
                dynamic sys = Py.Import("sys");
                Console.WriteLine("Python version: " + sys.version);
            }
        }

        private static void Shutdown()
        {
            if (!_disposed)
            {
                lock (_lock)
                {
                    if (!_disposed)
                    {
                        foreach (var p in _pipelines.Values)
                        {
                            p.Dispose();
                        }
                        _pipelines.Clear();
                        _disposed = true;
                        PythonEngine.EndAllowThreads(_threadState);
                        _threadState = IntPtr.Zero;

                        //The line below prints an error message on console ("release unlocked lock"), so removing for now
                        //PythonEngine.Shutdown();
                    }
                }
            }
        }
    }
}
