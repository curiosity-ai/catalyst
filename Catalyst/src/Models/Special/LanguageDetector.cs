using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO.Compression;

namespace Catalyst.Models
{
    public class LanguageDetectorModel : StorableObjectData
    {
        public Dictionary<int, Dictionary<Language, double>> WordLanguageProbabilities { get; set; } = new Dictionary<int, Dictionary<Language, double>>();
        public List<Language> Languages { get; set; } = new List<Language>();
    }

    public class LanguageDetector : StorableObject<LanguageDetector, LanguageDetectorModel>, IProcess
    {
        public double Alpha { get; set; } = 0.5;
        public int? RandomSeed { get; set; } = null;
        public int Trials { get; set; } = 7;
        public int NGramLength { get; set; } = 3;
        public int MaxTextLength { get; set; } = 10000;
        public double AlphaWidth { get; set; } = 0.05;
        public int MaxIterations { get; set; } = 1000;
        public double ProbabilityThreshold { get; set; } = 0.1;
        public double ConvergenceThreshold { get; set; } = 0.99999;
        public int BaseFrequency { get; set; } = 10000;

        public LanguageDetector(int version, string tag = "") : base(Language.Any, version, tag, compress: true)
        {
        }

        public new static async Task<LanguageDetector> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new LanguageDetector(version, tag);

            try
            {
                using var sr1 = typeof(LanguageDetector).Assembly.GetManifestResourceStream($"Catalyst.Resources.LanguageDetector.binz");
                using var decompressed = Storage.Current.GetTempStream();
                using (var ds = new DeflateStream(sr1, CompressionMode.Decompress, leaveOpen: true))
                {
                    await ds.CopyToAsync(decompressed);
                    decompressed.Seek(0, SeekOrigin.Begin);
                    a.Data = MessagePack.MessagePackSerializer.Deserialize<LanguageDetectorModel>(decompressed, Pipeline.LZ4Standard);
                    a.Version = 0;
                }
            }
            catch
            {
                await a.LoadDataAsync();
            }

            return a;
        }

        public void Process(IDocument document)
        {
            Detect(document);
        }

        public void Detect(IDocument doc)
        {
            doc.Language = Detect(doc.Value);
        }

        public Language Detect(string text)
        {
            var language = DetectAll(text).FirstOrDefault();
            return language is object ? language.Language : Language.Unknown;
        }

        public IEnumerable<DetectedLanguage> DetectAll(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) { return new DetectedLanguage[0]; }

            var ngrams = ExtractNGrams(NormalizeText(text.Substring(0, Math.Min(text.Length, 200))));

            if (ngrams.Count == 0) { return new DetectedLanguage[0]; }

            var languageProbabilities = new double[Data.Languages.Count];

            var random = RandomSeed is object ? new Random(RandomSeed.Value) : new Random();

            for (int t = 0; t < Trials; t++)
            {
                var probs = InitializeProbabilities();
                double alpha = Alpha + random.NextDouble() * AlphaWidth;

                for (int i = 0; ; i++)
                {
                    int r = random.Next(ngrams.Count);
                    UpdateProbabilities(probs, ngrams[r], alpha);
                    if (i % 5 == 0 && (NormalizeProbabilities(probs) > ConvergenceThreshold || i >= MaxIterations)) { break; }
                }

                for (int j = 0; j < languageProbabilities.Length; j++) { languageProbabilities[j] += probs[j] / Trials; }
            }

            return SortProbabilities(languageProbabilities);
        }

        private List<int> ExtractNGrams(string text)
        {
            var hashes = new List<int>();
            if (string.IsNullOrEmpty(text)) { return hashes; }
            NGram ngram = new NGram();
            foreach (char c in text)
            {
                ngram.Add(c);
                for (int n = 1; n <= NGram.N_GRAM; n++)
                {
                    string w = ngram.Get(n);

                    if (w is object)
                    {
                        int hash = GetHash(w);
                        if (Data.WordLanguageProbabilities.ContainsKey(hash))
                        {
                            hashes.Add(hash);
                        }
                    }
                }
            }

            return hashes;
        }

        public static int GetHash(string feature)
        {
            return Hashes.CaseSensitiveHash32(feature.Trim().ToLowerInvariant()); //TODO: Use hash from fastText
        }

        #region Normalize text

        private static readonly Regex UrlRegex = new Regex("https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}", RegexOptions.Compiled);
        private static readonly Regex EmailRegex = new Regex("[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}", RegexOptions.Compiled);

        private string NormalizeText(string text)
        {
            if (text.Length > MaxTextLength) { text = text.Substring(0, MaxTextLength); }

            text = RemoveAddresses(text);
            text = NormalizeAlphabet(text);
            text = NormalizeVietnamese(text);
            text = NormalizeWhitespace(text);

            return text;
        }

        private static string NormalizeAlphabet(string text)
        {
            int latinCount = 0;
            int nonLatinCount = 0;

            for (int i = 0; i < text.Length; ++i)
            {
                char c = text[i];

                if (c <= 'z' && c >= 'A')
                {
                    ++latinCount;
                }
                else if (c >= '\u0300' && !(c >= 0x1e00 && c <= 0x1eff))
                {
                    ++nonLatinCount;
                }
            }

            if (latinCount * 2 < nonLatinCount)
            {
                StringBuilder textWithoutLatin = new StringBuilder();
                for (int i = 0; i < text.Length; ++i)
                {
                    char c = text[i];
                    if (c > 'z' || c < 'A')
                        textWithoutLatin.Append(c);
                }
                text = textWithoutLatin.ToString();
            }

            return text;
        }

        private static string NormalizeVietnamese(string text)
        {
            // todo
            return text;
        }

        private static string NormalizeWhitespace(string text)
        {
            StringBuilder sb = new StringBuilder(text.Length);

            char? prev = null;

            foreach (char c in text)
            {
                if (c != ' ' || prev != ' ')
                    sb.Append(c);
                prev = c;
            }

            return sb.ToString();
        }

        private static string RemoveAddresses(string text)
        {
            text = UrlRegex.Replace(text, " ");
            text = EmailRegex.Replace(text, " ");
            return text;
        }

        #endregion Normalize text

        #region Probabilities

        private double[] InitializeProbabilities()
        {
            double[] prob = new double[Data.Languages.Count];
            for (int i = 0; i < prob.Length; i++) { prob[i] = 1.0 / Data.Languages.Count; }
            return prob;
        }

        private void UpdateProbabilities(double[] prob, int hashedWord, double alpha)
        {
            if (!Data.WordLanguageProbabilities.ContainsKey(hashedWord)) { return; }

            var languageProbabilities = Data.WordLanguageProbabilities[hashedWord];
            double weight = alpha / BaseFrequency;

            for (int i = 0; i < prob.Length; i++)
            {
                var profile = Data.Languages[i];
                prob[i] *= weight + (languageProbabilities.ContainsKey(profile) ? languageProbabilities[profile] : 0);
            }
        }

        private static double NormalizeProbabilities(double[] probs)
        {
            double maxp = 0, sump = 0;

            for (int i = 0; i < probs.Length; ++i) { sump += probs[i]; }
            for (int i = 0; i < probs.Length; ++i)
            {
                double p = probs[i] / sump;
                if (maxp < p) maxp = p;
                probs[i] = p;
            }
            return maxp;
        }

        private IEnumerable<DetectedLanguage> SortProbabilities(double[] probs)
        {
            List<DetectedLanguage> list = new List<DetectedLanguage>();

            for (int j = 0; j < probs.Length; j++)
            {
                double p = probs[j];

                if (p > ProbabilityThreshold)
                {
                    for (int i = 0; i <= list.Count; i++)
                    {
                        if (i == list.Count || list[i].Probability < p)
                        {
                            list.Insert(i, new DetectedLanguage { Language = Data.Languages[j], Probability = p });
                            break;
                        }
                    }
                }
            }

            return list;
        }

        #endregion Probabilities

        public class DetectedLanguage
        {
            public Language Language { get; set; }
            public double Probability { get; set; }
        }

        private class NGram
        {
            public const int N_GRAM = 3;

            private StringBuilder buffer = new StringBuilder(" ", N_GRAM);
            private bool capital = false;

            public void Add(char c)
            {
                char lastChar = buffer[buffer.Length - 1];

                if (lastChar == ' ')
                {
                    buffer = new StringBuilder(" ");
                    capital = false;
                    if (c == ' ') return;
                }
                else if (buffer.Length >= N_GRAM)
                {
                    buffer.Remove(0, 1);
                }

                buffer.Append(c);

                if (char.IsUpper(c))
                {
                    if (char.IsUpper(lastChar))
                        capital = true;
                }
                else
                {
                    capital = false;
                }
            }

            public string Get(int n)
            {
                if (capital)
                    return null;

                if (n < 1 || n > N_GRAM || buffer.Length < n)
                    return null;

                if (n == 1)
                {
                    char c = buffer[buffer.Length - 1];
                    if (c == ' ') return null;
                    return c.ToString();
                }
                else
                {
                    return buffer.ToString(buffer.Length - n, n);
                }
            }
        }

        private class JsonLanguageProfile
        {
            public string name = null;
            public Dictionary<string, int> freq = null;
            public int[] n_words = null;
        }

        public static void TransformJsonDataInModelData(string pathToFiles)
        {
            var files = Directory.GetFiles(pathToFiles);
            var ld = new LanguageDetector(version: 0);

            foreach (var f in files)
            {
                var language = Languages.CodeToEnum(Path.GetFileNameWithoutExtension(f));
                ld.Data.Languages.Add(language);

                string json = File.ReadAllText(f);
                var jsonProfile = JsonConvert.DeserializeObject<JsonLanguageProfile>(json);
                foreach (var word in jsonProfile.freq.Keys)
                {
                    int hash = GetHash(word);
                    if (!ld.Data.WordLanguageProbabilities.ContainsKey(hash))
                    {
                        ld.Data.WordLanguageProbabilities[hash] = new Dictionary<Language, double>();
                    }

                    if (word.Length >= 1 && word.Length <= ld.NGramLength)
                    {
                        double prob = (double)jsonProfile.freq[word] / jsonProfile.n_words[word.Length - 1];
                        ld.Data.WordLanguageProbabilities[hash][language] = prob;
                    }
                }
            }

            ld.StoreAsync().Wait();
        }
    }
}