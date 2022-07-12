using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class SpotterModel : StorableObjectData
    {
        public HashSet<ulong> Hashes { get; set; } = new HashSet<ulong>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public string CaptureTag { get; set; }
        public Dictionary<int, TokenizationException> TokenizerExceptions { get; set; } = new Dictionary<int, TokenizationException>();
        public bool IgnoreOnlyNumeric { get; set; }
        public bool IgnoreCase { get; set; }
    }

    public class Spotter : StorableObjectV2<Spotter, SpotterModel>, IEntityRecognizer, IProcess, IHasSpecialCases
    {
        public string CaptureTag => Data.CaptureTag;

        public bool IgnoreCase { get { return Data.IgnoreCase; } set { Data.IgnoreCase = value; } }

        public const string Separator = "_";

        private Spotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public Spotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        public new static async Task<Spotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new Spotter(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { CaptureTag };
        }

        public bool RecognizeEntities(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span);
            }
            return foundAny;
        }

        public bool HasAnyEntity(IDocument document)
        {
            foreach (var span in document)
            {
                if (RecognizeEntities(span, stopOnFirstFound: true))
                {
                    return true;
                }
            }
            return false;
        }

        public bool IsEquivalentTo(Spotter other)
        {
            var omd = other.Data;
            var tmd = this.Data;
            return omd.IgnoreOnlyNumeric == tmd.IgnoreOnlyNumeric &&
                   omd.IgnoreCase == tmd.IgnoreCase &&
                   omd.Hashes.SetEquals(tmd.Hashes) &&
                   omd.MultiGramHashes.Count == tmd.MultiGramHashes.Count &&
                   omd.MultiGramHashes.Zip(tmd.MultiGramHashes, (a, b) => a.SetEquals(b)).All(b => b);
        }

        public static ulong HashCombine64(ulong rhs, ulong lhs)
        {
            lhs ^= rhs + 0x9e3779b97f492000 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Hash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += key[i];
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong IgnoreCaseHash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += char.ToLowerInvariant(key[i]);
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        public void ClearModel()
        {
            Data.Hashes.Clear();
            Data.MultiGramHashes.Clear();
            Data.TokenizerExceptions.Clear();
        }

        public bool RecognizeEntities(ISpan ispan, bool stopOnFirstFound = false)
        {
            var tokens = ispan.ToTokenSpan();
            int N = tokens.Length;
            bool hasMultiGram = Data.MultiGramHashes.Any();
            bool foundAny = false;
            for (int i = 0; i < N; i++)
            {
                var tk = tokens[i];
                //if (tk.POS != PartOfSpeechEnum.NOUN && tk.POS != PartOfSpeechEnum.ADJ && tk.POS != PartOfSpeechEnum.PROPN) { continue; }

                var tokenHash = Data.IgnoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                if (hasMultiGram && Data.MultiGramHashes[0].Contains(tokenHash))
                {
                    int window = Math.Min(N - i, Data.MultiGramHashes.Count);
                    ulong hash = tokenHash;
                    bool someTokenHasReplacements = tk.Replacement is object;
                    int i_final = i;

                    for (int n = 1; n < window; n++)
                    {
                        var next = tokens[n + i];
                        someTokenHasReplacements |= (next.Replacement is object);

                        var nextHash = Data.IgnoreCase ? IgnoreCaseHash64(next.ValueAsSpan) : Hash64(next.ValueAsSpan);
                        if (Data.MultiGramHashes[n].Contains(nextHash))
                        {
                            hash = HashCombine64(hash, nextHash);
                            if (Data.Hashes.Contains(hash))
                            {
                                i_final = i + n;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (i_final > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                        tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Begin));
                        tokens[i_final].AddEntityType(new EntityType(CaptureTag, EntityTag.End));

                        for (int m = i + 1; m < (i_final); m++)
                        {
                            tokens[m].AddEntityType(new EntityType(CaptureTag, EntityTag.Inside));
                        }
                    }

                    i = i_final;
                }

                if (Data.Hashes.Contains(tokenHash))
                {
                    foundAny = true;
                    if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                    tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Single));
                }
            }
            return foundAny;
        }

        private ReaderWriterLockSlim TrainLock = new ReaderWriterLockSlim();

        public void TrainWord2Sense(IEnumerable<IDocument> documents, ParallelOptions parallelOptions, int ngrams = 3, double tooRare = 1E-5, double tooCommon = 0.1, Word2SenseTrainingData trainingData = null)
        {
            var hashCount          = new ConcurrentDictionary<ulong, int>(trainingData?.HashCount           ?? new Dictionary<ulong, int>());
            var senses             = new ConcurrentDictionary<ulong, ulong[]>(trainingData?.Senses          ?? new Dictionary<ulong, ulong[]>());
            var words              = new ConcurrentDictionary<ulong, string>(trainingData?.Words            ?? new Dictionary<ulong, string>());
            var shapes             = new ConcurrentDictionary<string, ulong>(trainingData?.Shapes           ?? new Dictionary<string, ulong>());
            var shapeExamples      = new ConcurrentDictionary<string, string[]>(trainingData?.ShapeExamples ?? new Dictionary<string, string[]>());

            long totalDocCount     = trainingData?.SeenDocuments ?? 0;
            long totalTokenCount   = trainingData?.SeenTokens ?? 0;

            bool ignoreCase        = Data.IgnoreCase;
            bool ignoreOnlyNumeric = Data.IgnoreOnlyNumeric;
            var stopwords          = new HashSet<ulong>(StopWords.Spacy.For(Language).Select(w => ignoreCase ? IgnoreCaseHash64(w.AsSpan()) : Hash64(w.AsSpan())).ToArray());

            int docCount = 0, tkCount = 0;

            var sw = Stopwatch.StartNew();

            TrainLock.EnterWriteLock();
            try
            {
                Parallel.ForEach(documents, parallelOptions, doc =>
                {
                    try
                    {
                        var stack = new Queue<ulong>(ngrams);

                        if (doc.TokensCount < ngrams) { return; } //Ignore too small documents

                        Interlocked.Add(ref tkCount, doc.TokensCount);
                        
                        foreach (var span in doc)
                        {
                            var tokens = span.GetCapturedTokens().ToArray();

                            for (int i = 0; i < tokens.Length; i++)
                            {
                                var tk = tokens[i];

                                if (!(tk is Tokens))
                                {
                                    var shape = tk.ValueAsSpan.Shape(compact: false);
                                    shapes.AddOrUpdate(shape, 1, (k, v) => v + 1);

                                    shapeExamples.AddOrUpdate(shape, (k) => new[] { tk.Value }, (k, v) =>
                                    {
                                        if (v.Length < 50)
                                        {
                                            v = v.Concat(new[] { tk.Value }).Distinct().ToArray();
                                        }
                                        return v;
                                    });
                                }

                                var hash = ignoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                                bool filterPartOfSpeech = !(tk.POS == PartOfSpeech.ADJ || tk.POS == PartOfSpeech.NOUN);

                                bool skipIfHasUpperCase = (!ignoreCase && !tk.ValueAsSpan.IsAllLowerCase());

                                bool skipIfTooSmall = (tk.Length < 3);

                                bool skipIfNotAllLetterOrDigit = !(tk.ValueAsSpan.IsAllLetterOrDigit());

                                bool skipIfStopWordOrEntity = stopwords.Contains(hash) || tk.EntityTypes.Any();

                                //Heuristic for ordinal numbers (i.e. 1st, 2nd, 33rd, etc)
                                bool skipIfMaybeOrdinal = (tk.ValueAsSpan.IndexOfAny(new char[] { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 't', 'h', 's', 't', 'r', 'd' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 'a', 'b', 'c', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z' }, 0) < 0);

                                bool skipIfOnlyNumeric = ignoreOnlyNumeric ? !tk.ValueAsSpan.IsLetter() : false;

                                //Only filter for POS if language != any, as otherwise we won't have the POS information
                                bool skipThisToken = (filterPartOfSpeech && Language != Language.Any) || skipIfHasUpperCase || skipIfTooSmall || skipIfNotAllLetterOrDigit || skipIfStopWordOrEntity || skipIfMaybeOrdinal || skipIfOnlyNumeric;

                                if (skipThisToken)
                                {
                                    stack.Clear();
                                    continue;
                                }

                                if (!words.ContainsKey(hash)) { words[hash] = ignoreCase ? tk.Value.ToLowerInvariant() : tk.Value; }

                                stack.Enqueue(hash);
                                ulong combined = stack.ElementAt(0);

                                for (int j = 1; j < stack.Count; j++)
                                {
                                    combined = HashCombine64(combined, stack.ElementAt(j));
                                    if (hashCount.ContainsKey(combined))
                                    {
                                        hashCount[combined]++;
                                    }
                                    else
                                    {
                                        senses[combined] = stack.Take(j + 1).ToArray();
                                        hashCount[combined] = 1;
                                    }
                                }

                                if (stack.Count > ngrams) { stack.Dequeue(); }
                            }
                        }

                        int count = Interlocked.Increment(ref docCount);

                        if (count % 1000 == 0)
                        {
                            Logger.LogInformation("Training Word2Sense model - at {DOCCOUNT} documents, {TKCOUNT} tokens - elapsed {ELAPSED} seconds at {KTKS} kTk/s)", docCount, tkCount, sw.Elapsed.TotalSeconds, (tkCount / sw.ElapsedMilliseconds));
                        }
                    }
                    catch (Exception E)
                    {
                        Logger.LogError(E, "Error during training Word2Sense model");
                    }
                });
            }
            catch (OperationCanceledException)
            {
                return;
            }
            finally
            {
                TrainLock.ExitWriteLock();
            }

            Logger.LogInformation("Finish parsing documents for Word2Sense model");

            totalDocCount += docCount;
            totalTokenCount += tkCount;

            int thresholdRare   = Math.Max(2, (int)Math.Floor(tooRare * totalTokenCount));
            int thresholdCommon = (int)Math.Floor(tooCommon * totalTokenCount);

            var toKeep = hashCount.Where(kv => kv.Value >= thresholdRare && kv.Value <= thresholdCommon).OrderByDescending(kv => kv.Value)
                                                .Select(kv => kv.Key).ToArray();

            foreach (var key in toKeep)
            {
                if (senses.TryGetValue(key, out var hashes) && hashCount.TryGetValue(key, out var count))
                {
                    Data.Hashes.Add(key);
                    for (int i = 0; i < hashes.Length; i++)
                    {
                        if (Data.MultiGramHashes.Count <= i)
                        {
                            Data.MultiGramHashes.Add(new HashSet<ulong>());
                        }
                        Data.MultiGramHashes[i].Add(hashes[i]);
                    }
                }
            }

            if(trainingData is object)
            {
                trainingData.HashCount = new Dictionary<ulong, int>(hashCount);
                trainingData.Senses = new Dictionary<ulong, ulong[]>(senses);
                trainingData.Words = new Dictionary<ulong, string>(words);
                trainingData.SeenDocuments = totalDocCount;
                trainingData.SeenTokens = totalTokenCount;
                trainingData.Shapes = new Dictionary<string, ulong>(shapes);
                trainingData.ShapeExamples = new Dictionary<string, string[]>(shapeExamples);
            }

            Logger.LogInformation("Finish training Word2Sense model");
        }

        public IEnumerable<KeyValuePair<int, TokenizationException>> GetSpecialCases()
        {
            if (Data.TokenizerExceptions is object)
            {
                foreach (var sc in Data.TokenizerExceptions)
                {
                    yield return sc;
                }
            }
        }

        public void AddEntry(string entry)
        {
            void AddSingleTokenConcept(ulong entryHash)
            {
                Data.Hashes.Add(entryHash);
            }

            if (string.IsNullOrWhiteSpace(entry)) { return; }


            if (Data.IgnoreOnlyNumeric && int.TryParse(entry, out _)) { return; } //Ignore pure numerical entries

            var words = entry.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 1)
            {
                var hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[0].AsSpan()) : Spotter.Hash64(words[0].AsSpan());
                AddSingleTokenConcept(hash);

                if (!words[0].AsSpan().IsLetter())
                {
                    Data.TokenizerExceptions[words[0].CaseSensitiveHash32()] = new TokenizationException(null); //Null means don't replace by anything - keep token as is
                }

                return;
            }

            ulong combinedHash = 0;
            for (int n = 0; n < words.Length; n++)
            {
                var word_hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[n].AsSpan()) : Spotter.Hash64(words[n].AsSpan());
                if (n == 0) { combinedHash = word_hash; } else { combinedHash = Spotter.HashCombine64(combinedHash, word_hash); }
                if (Data.MultiGramHashes.Count < n + 1)
                {
                    Data.MultiGramHashes.Add(new HashSet<ulong>());
                }

                if (!Data.MultiGramHashes[n].Contains(word_hash))
                {
                    Data.MultiGramHashes[n].Add(word_hash);
                }

                if (!words[n].AsSpan().IsLetter())
                {
                    Data.TokenizerExceptions[words[n].CaseSensitiveHash32()] = new TokenizationException(null); //Null means don't replace by anything - keep token as is
                }
            }

            AddSingleTokenConcept(combinedHash);
        }

        public void AppendList(IEnumerable<string> words)
        {
            foreach (var word in words)
            {
                AddEntry(word);
            }
        }
    }

    [MessagePack.MessagePackObject(keyAsPropertyName: true)]
    public class Word2SenseTrainingData
    {
        public Dictionary<ulong, int> HashCount { get; set; } = new Dictionary<ulong, int>();
        public Dictionary<ulong, ulong[]> Senses { get; set; } = new Dictionary<ulong, ulong[]>();
        public Dictionary<ulong, string> Words { get; set; } = new Dictionary<ulong, string>();
        public Dictionary<string, ulong> Shapes { get; set; } = new Dictionary<string, ulong>();
        public Dictionary<string, string[]> ShapeExamples { get; set; } = new Dictionary<string, string[]>();
        public long SeenDocuments { get; set; } = 0;
        public long SeenTokens { get; set; } = 0;
    }
}