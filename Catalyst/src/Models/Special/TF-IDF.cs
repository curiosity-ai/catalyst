using Mosaik.Core;

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class TFIDFModel : StorableObjectData
    {
        public Dictionary<int, int> Word_DF { get; set; } = new Dictionary<int, int>();
        public int DocumentCount;
    }

    public class TFIDFToken
    {
        public int WordHash;
        public double IDF;

        public TFIDFToken(IToken source, int wordHash)
        {
            Source = source; WordHash = wordHash;
        }

        public IToken Source { get; set; }
        public ITokens Sources { get; set; }

        public TFIDFToken(ITokens sources, int wordHash)
        {
            Sources = sources; WordHash = wordHash;
        }
    }

    public class TFIDF_Trainer : TFIDF, IProcess
    {
        public TFIDF_Trainer(Language language, int version, string tag) : base(language, version, tag)
        {
            Data = new TFIDFModel();
            _WordDF = new ConcurrentDictionary<int, int>();
        }

        public new void Process(IDocument document)
        {
            UpdateVocabulary(ExtractTokenHashes(document));
        }

        public override async Task StoreAsync()
        {
            int minCount = (int)(_MinimumDF * _DocumentCount);
            Data.Word_DF = _WordDF.Where(tk => tk.Value > minCount).ToDictionary(tk => tk.Key, tk => tk.Value);
            Data.DocumentCount = _DocumentCount;
            await base.StoreAsync();
        }
    }

    public class TFIDF : StorableObject<TFIDF, TFIDFModel>, IProcess
    {
        internal object _VocabularyLock = new object();
        internal ConcurrentDictionary<int, int> _WordDF { get; set; }
        internal int _DocumentCount = 0;
        internal double _MinimumDF = 0.0001; //0.01 % of documents

        public TFIDF(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public new static async Task<TFIDF> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new TFIDF(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Compute(IDocument doc)
        {
            //(var monograms, var bigrams, var trigrams) = ExtractTokens(doc);
            //var alltokens = monograms.Union(bigrams).Union(trigrams);
            var tokens = ExtractTokens(doc);
            var tf = tokens.GroupBy(tk => tk.WordHash).ToDictionary(g => g.Key, g => g.Count());
            var tf_idf = tf.ToDictionary(kv => kv.Key, kv => (1.0 + Math.Log10(kv.Value)) * GetWordIDF(kv.Key));

            //The order comes from ExtractToken: single tokens are processed first, then their frequency is replaced with the entities frequency
            foreach (var token in tokens)
            {
                if (token.Source is object)
                {
                    token.Source.Frequency = (float)tf_idf[token.WordHash];
                }
                else if (token.Sources is object)
                {
                    token.Sources.Children.ToList().ForEach(tk => tk.Frequency = (float)tf_idf[token.WordHash]);
                }
            }
        }

        private double GetWordIDF(int key)
        {
            int DF = 0;
            Data.Word_DF.TryGetValue(key, out DF);
            return Math.Log(Data.DocumentCount / (1.0 + DF));
        }

        public async Task Train(IEnumerable<IDocument> documents)
        {
            Data = new TFIDFModel();
            _WordDF = new ConcurrentDictionary<int, int>();

            documents.AsParallel().ForAll(doc => UpdateVocabulary(ExtractTokenHashes(doc)));

            Data.Word_DF = _WordDF.ToDictionary(tk => tk.Key, tk => tk.Value);
            Data.DocumentCount = documents.Count();

            await StoreAsync();
        }

        internal static IList<int> ExtractTokenHashes(IDocument doc)
        {
            var hashes = doc.SelectMany(s => s.Tokens).Select(t => t.IgnoreCaseHash).ToList();
            hashes.AddRange(doc.SelectMany(s => s.GetEntities()).Select(e => e.IgnoreCaseHash));
            return hashes;
        }

        internal static List<TFIDFToken> ExtractTokens(IDocument doc)
        {
            //The order here is important, as single tokens are processed first, then their frequency is replaced with the entities frequency
            var tokens = doc.SelectMany(s => s.Tokens).Select(t => new TFIDFToken(t, t.IgnoreCaseHash)).ToList();
            var entities = doc.SelectMany(s => s.GetEntities()).Select(e => new TFIDFToken(e, e.IgnoreCaseHash));
            tokens.AddRange(entities);
            return tokens;
        }

        internal void UpdateVocabulary(IEnumerable<int> alltokens)
        {
            lock (_VocabularyLock)
            {
                var wordhashes = alltokens.Distinct();
                _DocumentCount++;
                foreach (var h in wordhashes) { _WordDF.AddOrUpdate(h, 1, (key, oldValue) => oldValue + 1); }
            }
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Compute(document);
        }
    }

    internal class TFIDFTokenComparer : IEqualityComparer<TFIDFToken>
    {
        public TFIDFTokenComparer()
        {
        }

        public bool Equals(TFIDFToken x, TFIDFToken y)
        {
            return (x.WordHash == y.WordHash);
        }

        public int GetHashCode(TFIDFToken obj)
        {
            return obj.WordHash;
        }
    }
}