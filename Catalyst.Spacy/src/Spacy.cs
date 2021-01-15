using Mosaik.Core;
using Python.Deployment;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;

namespace Catalyst
{
    public static class Spacy
    {
        public enum ModelSize
        {
            Small,
            Medium,
            Large
        }

        private const string DownloadURL = "https://github.com/explosion/spacy-models/releases/download";
        private const string CompatibilityDataURL = "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json";
        private const string ShortcutsURL = "https://raw.githubusercontent.com/explosion/spacy-models/master/shortcuts-v2.json";

        public static Dictionary<string, Dictionary<string, Dictionary<string, string[]>>> CompatibilityData { get; private set; }
        public static Dictionary<string, string> Shortcuts { get; private set; }
        public static string SpacyVersion { get; private set; }

        public static async Task<PythonLock> Initialize(ModelSize modelSize, params Language[] languages)
        {
            await Installer.SetupPython();
            Installer.TryInstallPip();
            Installer.PipInstallModule("spacy");
            Installer.PipInstallModule("spacy-lookups-data");
            PythonEngine.Initialize();

            var threadState = PythonEngine.BeginAllowThreads();

            TestPythonVersion();

            await LoadModelsDataAsync();

            LoadModelsData(modelSize, languages);

            return new PythonLock(threadState);
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
            modelName += $"_{ size}";
            return modelName;
        }

        private static string GetSpacyVersion()
        {
            using (Py.GIL())
            {
                dynamic spacy = Py.Import("spacy");
                Console.WriteLine("Spacy version: " + spacy.__version__);

                return spacy.__version__;
            }
        }

        private static void TestPythonVersion()
        {
            using (Py.GIL())
            {
                dynamic sys = PythonEngine.ImportModule("sys");
                Console.WriteLine("Python version: " + sys.version);
            }
        }

        public sealed class PythonLock : IDisposable
        {
            private bool _disposed = false;
            private IntPtr _threadState;
            private readonly object _lock = new object();

            public PythonLock(IntPtr threadState)
            {
                _threadState = threadState;
            }

            public void Dispose()
            {
                if (!_disposed)
                {
                    lock (_lock)
                    {
                        if (!_disposed) 
                        {
                            _disposed = true;
                            PythonEngine.EndAllowThreads(_threadState);
                            PythonEngine.Shutdown();
                        }
                    }
                }
            }
        }

        public sealed class SpacyPipeline
        {
            private dynamic _nlp;

            public Language Language { get; }
            public ModelSize ModelSize { get; }

            internal SpacyPipeline(Language language, ModelSize modelSize, dynamic pipeline)
            {
                _nlp = pipeline;
                Language = language;
                ModelSize = modelSize;
            }

            public void ProcessSingle(Document document)
            {
                using (Py.GIL())
                {
                    var s_doc = _nlp(document.Value);
                    SyncBack(s_doc, document);
                }
            }

            public void Process(IEnumerable<Document> documents)
            {
                var batch = new List<Document>();
                foreach(var doc in documents)
                {
                    batch.Add(doc);

                    if(batch.Count > 1000)
                    {
                        ProcessBatch(batch);
                        batch.Clear();
                    }
                }

                ProcessBatch(batch);

                void ProcessBatch(List<Document> docs)
                {
                    if (docs.Count > 0)
                    {
                        using (Py.GIL())
                        {
                            var s_docs = _nlp.pipe(docs.Select(d => d.Value).ToArray());

                            for (int i = 0; i < docs.Count; i++)
                            {
                                SyncBack(s_docs[i], docs[i]);
                            }
                        }
                    }
                }
            }

            private void SyncBack(dynamic s_doc, Document document)
            {
                foreach (var s_sentence in s_doc.sents)
                {
                    var span = document.AddSpan((int)s_sentence.start_char, (int)s_sentence.end_char);
                    foreach (var s_token in s_sentence)
                    {
                        var tb = (int)s_token.idx;
                        var token = span.AddToken(tb, tb + (int)s_token.__len__() - 1);


                        token.POS = ConvertPOS((string)s_token.pos_);
                        token.DependencyType = (string)s_token.dep_;

                        var head = s_token.head;
                        if (head is object)
                        {
                            token.Head = (int)head.i;
                        }
                        else
                        {
                            token.Head = -1;
                        }
                    }
                }

                //for ent in doc.ents: print(ent.text, ent.start_char, ent.end_char, ent.label_)
            }

            private PartOfSpeech ConvertPOS(string s_pos)
            {
                return Enum.TryParse<PartOfSpeech>(s_pos, out var pos) ? pos : PartOfSpeech.X;
            }
        }

        private static Dictionary<string, SpacyPipeline> _pipelines = new Dictionary<string, SpacyPipeline>();

        public static SpacyPipeline Pipeline(ModelSize modelSize, Language language)
        {
            using (Py.GIL())
            {
                var key = $"{language}-{modelSize}";

                if (_pipelines.TryGetValue(key, out var pipeline)) return pipeline;

                dynamic spacy = Py.Import("spacy");
                var modelName = GetModelName(modelSize, language);
                var nlp = spacy.load(modelName);
                pipeline = new SpacyPipeline(language, modelSize, nlp);

                _pipelines[key] = pipeline;

                return pipeline;
            }
        }
    }
}
