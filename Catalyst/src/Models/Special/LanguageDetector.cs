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
using System.Threading;

namespace Catalyst.Models
{
    public class LanguageDetectorModel : StorableObjectData
    {
        public Dictionary<int, Dictionary<Language, double>> WordLanguageProbabilities { get; set; } = new Dictionary<int, Dictionary<Language, double>>();
        public List<Language> Languages { get; set; } = new List<Language>();
    }

    public class LanguageDetector : StorableObject<LanguageDetector, LanguageDetectorModel>, IProcess
    {
        private static ObjectPool<List<int>> _listPool = new ObjectPool<List<int>>(() => new List<int>(), Environment.ProcessorCount, l => l.Clear());
        public double Alpha { get; set; } = 0.5;
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
                using var decompressed = new MemoryStream();
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

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Detect(document);
        }

        public void Detect(IDocument doc)
        {
            doc.Language = Detect(doc.Value);
        }

        public Language Detect(string text)
        {
            var languages = DetectAll(text);
            return languages.Count > 0 ? languages[0].Language : Language.Unknown;
        }

        public IList<DetectedLanguage> DetectAll(string text)
        {
            if (string.IsNullOrWhiteSpace(text)) { return Array.Empty<DetectedLanguage>(); }

            var ngrams = ExtractNGrams(NormalizeText(text));

            try
            {
                if (ngrams.Count == 0) { return Array.Empty<DetectedLanguage>(); }

                Span<double> languageProbabilities = Data.Languages.Count < 256 ? stackalloc double[Data.Languages.Count] : new double[Data.Languages.Count];

                for (int t = 0; t < Trials; t++)
                {
                    var probs = InitializeProbabilities();
                    
                    double alpha = Alpha + ThreadSafeFastRandom.NextDouble() * AlphaWidth;

                    for (int i = 0; ; i++)
                    {
                        int r = ThreadSafeFastRandom.Next(ngrams.Count);
                        UpdateProbabilities(probs, ngrams[r], alpha);
                        if (i % 5 == 0 && (NormalizeProbabilities(probs) > ConvergenceThreshold || i >= MaxIterations)) { break; }
                    }

                    for (int j = 0; j < languageProbabilities.Length; j++) { languageProbabilities[j] += probs[j] / Trials; }
                }

                return SortProbabilities(languageProbabilities);
            }
            finally
            {
                _listPool.Return(ngrams);
            }
        }

        private List<int> ExtractNGrams(string text)
        {
            var hashes = _listPool.Rent();
            if (string.IsNullOrEmpty(text)) { return hashes; }
            
            var ngram = new NGram();

            foreach (char c in text)
            {
                ngram.Add(c);

                for (int n = 1; n <= NGram.N_GRAM; n++)
                {
                    var w = ngram.Get(n);

                    if (w.Length > 0)
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

        private static int GetHash(ReadOnlySpan<char> lowerCaseFeaturesWithoutSpaces)
        {
            return Hashes.CaseSensitiveHash32(lowerCaseFeaturesWithoutSpaces);
        }

        #region Normalize text

        private static readonly Regex RE_Numbers = new Regex(@"([\w+\-!?\\\/\(\)\[\]\{\},.:;=$%&]*\d[\w\d+\-!?\\\/\(\)\[\]\{\},.:;=$%&]+)", RegexOptions.Compiled);
        private static readonly Regex UrlRegex = new Regex("https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}", RegexOptions.Compiled);
        private static readonly Regex EmailRegex = new Regex("[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}", RegexOptions.Compiled);

        private string NormalizeText(string text)
        {
            text = NormalizeWhitespace(text);

            if (text.Length > MaxTextLength) { text = text.Substring(0, MaxTextLength); }

            text = RemoveAddresses(text);
            text = RemoveNumbers(text);
            text = NormalizeAlphabet(text);
            text = NormalizeVietnamese(text);
            text = NormalizeWhitespace(text);

            return text;
        }

        private string RemoveNumbers(string text)
        {
            return RE_Numbers.Replace(text, " ");
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
            var sb = Pools.StringBuilder.Rent();

            bool prevIsSpace = false;

            foreach (char c in text)
            {
                if(char.IsWhiteSpace(c))
                {
                    if(!prevIsSpace)
                    {
                        sb.Append(' ');
                    }

                    prevIsSpace = true;
                }
                else
                {
                    sb.Append(c);
                    prevIsSpace = false;
                }
            }

            var final = sb.ToString();
            
            Pools.StringBuilder.Return(sb);

            return final;
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

        private IList<DetectedLanguage> SortProbabilities(Span<double> probs)
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

        public struct DetectedLanguage
        {
            public Language Language { get; set; }
            public double Probability { get; set; }
        }

        private class NGram
        {
            public const int N_GRAM = 3;

            private readonly char[] buffer;
            private int length;

            private bool capital = false;

            public NGram()
            {
                buffer = new char[N_GRAM];
                buffer[0] = ' ';
                length = 1;
            }

            public void Add(char c)
            {
                char lastChar = buffer[length - 1];

                if (lastChar == ' ')
                {
                    buffer[0] = ' ';
                    length = 1;

                    capital = false;
                    if (c == ' ') return;
                }
                else if (length == N_GRAM)
                {
                    buffer[0] = buffer[1];
                    buffer[1] = buffer[2];
                    length = 2;
                }

                buffer[length] = char.ToLowerInvariant(c);
                length++;

                if (char.IsUpper(c))
                {
                    if (char.IsUpper(lastChar))
                    {
                        capital = true;
                    }
                }
                else
                {
                    capital = false;
                }
            }

            public ReadOnlySpan<char> Get(int n)
            {
                if (capital) return ReadOnlySpan<char>.Empty;

                if (n < 1 || n > N_GRAM || length < n) return ReadOnlySpan<char>.Empty;

                if (n == 1)
                {
                    char c = buffer[length - 1];
                    
                    if (c == ' ') return ReadOnlySpan<char>.Empty;

                    return Trim(buffer.AsSpan(length - 1, 1));
                }
                else
                {
                    return Trim(buffer.AsSpan(length - n, n));
                }
            }

            private ReadOnlySpan<char> Trim(Span<char> span)
            {
                if (span.Length == 0) return span;
                bool allSpace = true;
                foreach(var c in span) { allSpace &= char.IsWhiteSpace(c); }
                if (allSpace) return ReadOnlySpan<char>.Empty;

                var s = 0;
                var e = span.Length-1;
                for (int i = 0; i < span.Length; i++)
                {
                    if (char.IsWhiteSpace(span[i]))
                    {
                        s = i;
                    }
                    else
                    {
                        break;
                    }
                }

                for (int i = span.Length-1; i > s; i--)
                {
                    if (char.IsWhiteSpace(span[i]))
                    {
                        e = i;
                    }
                    else
                    {
                        break;
                    }
                }

                return span.Slice(s, e + 1);
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
                    int hash = GetHash(word.ToLowerInvariant().Trim().AsSpan());
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