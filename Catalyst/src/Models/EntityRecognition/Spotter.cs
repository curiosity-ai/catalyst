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

namespace Catalyst.Models
{
    public class SpotterModel : StorableObjectData
    {
        public HashSet<ulong> Hashes { get; set; } = new HashSet<ulong>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public string CaptureTag { get; set; }
        public Dictionary<int, TokenizationException> TokenizerExceptions { get; set; } = new Dictionary<int, TokenizationException>();
        public bool IgnoreOnlyNumeric { get; set; }

        #region IgnoreCaseFix

        // This fixes the mistake made in the naming of this variable (invariant case != ignore case).
        // As we cannot rename here (due to the serialization using keyAsPropertyName:true), we add a second property
        // that refers to the same underlying variable. As MessagePack reads properties in the order of GetProperties,
        // this ensures the new one (IgnoreCase) is set before the old one (InvariantCase), so we don't the stored value
        private bool ignoreCase;

        public bool IgnoreCase { get { return ignoreCase; } set { ignoreCase = value; } }

        [Obsolete("Wrong property name, use IgnoreCase instead", true)]
        public bool InvariantCase { get { return ignoreCase; } set { ignoreCase = value; } }

        #endregion IgnoreCaseFix
    }

    public class Spotter : StorableObject<Spotter, SpotterModel>, IEntityRecognizer, IProcess, IHasSpecialCases
    {
        public string CaptureTag => Data.CaptureTag;

        public bool IgnoreCase { get { return Data.IgnoreCase; } set { Data.IgnoreCase = value; } }

        public const string Separator = "_";

        public List<string> TempGazeteer = new List<string>();

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

        public void Process(IDocument document)
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
            var tmd = Data;
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
                            //txt += " " + next.Value;
                            //var hashTxt = Hash64(txt);
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
            if (trainingData == null)
            {
                trainingData = new Word2SenseTrainingData();
            }

            var stopwords = new HashSet<ulong>(StopWords.Spacy.For(Language).Select(w => Data.IgnoreCase ? IgnoreCaseHash64(w.AsSpan()) : Hash64(w.AsSpan())).ToArray());

            int docCount = 0, tkCount = 0;

            var sw = Stopwatch.StartNew();

            TrainLock.EnterWriteLock();
            try
            {
                Parallel.ForEach(documents, parallelOptions, doc =>
                {
                    try
                    {
                        var Previous = new ulong[ngrams];
                        var Stack = new Queue<ulong>(ngrams);

                        if (doc.TokensCount < ngrams) { return; } //Ignore too small documents

                        Interlocked.Add(ref tkCount, doc.TokensCount);
                        foreach (var span in doc)
                        {
                            var tokens = span.GetTokenized().ToArray();

                            for (int i = 0; i < tokens.Length; i++)
                            {
                                var tk = tokens[i];

                                var hash = Data.IgnoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                                bool filterPartOfSpeech = !(tk.POS == PartOfSpeech.ADJ || tk.POS == PartOfSpeech.NOUN || tk.POS == PartOfSpeech.PROPN);

                                bool skipIfHasUpperCase = (!Data.IgnoreCase && !tk.ValueAsSpan.IsAllLowerCase());

                                bool skipIfTooSmall = (tk.Length < 3);

                                bool skipIfNotAllLetterOrDigit = !(tk.ValueAsSpan.IsAllLetterOrDigit());

                                bool skipIfStopWordOrEntity = stopwords.Contains(hash) || tk.EntityTypes.Any();

                                //Heuristic for ordinal numbers (i.e. 1st, 2nd, 33rd, etc)
                                bool skipIfMaybeOrdinal = (tk.ValueAsSpan.IndexOfAny(new char[] { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 't', 'h', 's', 't', 'r', 'd' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 'a', 'b', 'c', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z' }, 0) < 0);

                                bool skipThisToken = filterPartOfSpeech || skipIfHasUpperCase || skipIfTooSmall || skipIfNotAllLetterOrDigit || skipIfStopWordOrEntity || skipIfMaybeOrdinal;

                                if (skipThisToken)
                                {
                                    Stack.Clear();
                                    continue;
                                }

                                if (!trainingData.Words.ContainsKey(hash)) { trainingData.Words[hash] = Data.IgnoreCase ? tk.Value.ToLowerInvariant() : tk.Value; }

                                Stack.Enqueue(hash);
                                ulong combined = Stack.ElementAt(0);

                                for (int j = 1; j < Stack.Count; j++)
                                {
                                    combined = HashCombine64(combined, Stack.ElementAt(j));
                                    if (trainingData.HashCount.ContainsKey(combined))
                                    {
                                        trainingData.HashCount[combined]++;
                                    }
                                    else
                                    {
                                        trainingData.Senses[combined] = Stack.Take(j + 1).ToArray();
                                        trainingData.HashCount[combined] = 1;
                                    }
                                }

                                if (Stack.Count > ngrams) { Stack.Dequeue(); }
                            }
                        }

                        int count = Interlocked.Increment(ref docCount);

                        if (count % 1000 == 0)
                        {
                            var mem = GC.GetTotalMemory(false);
                            Logger.LogInformation("[MEM: {MEMORY} MB]  Training Word2Sense model - at {DOCCOUNT} documents, {TKCOUNT} tokens - elapsed {ELAPSED} seconds at {KTKS} kTk/s)", Math.Round(mem / 1048576.0, 2), docCount, tkCount, sw.Elapsed.TotalSeconds, (tkCount / sw.ElapsedMilliseconds));
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

            int thresholdRare = (int)Math.Floor(tooRare * docCount);
            int thresholdCommon = (int)Math.Floor(tooCommon * docCount);

            var toKeep = trainingData.HashCount.Where(kv => kv.Value >= thresholdRare && kv.Value <= thresholdCommon).OrderByDescending(kv => kv.Value)
                                  .Select(kv => kv.Key).ToArray();

            foreach (var key in toKeep)
            {
                var hashes = trainingData.Senses[key];
                var count = trainingData.HashCount[key];

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

        public void AppendList(IEnumerable<string> words)
        {
            foreach (var word in words)
            {
                var w = word.Trim();

                if (w.Contains(' '))
                {
                    var parts = w.Split(CharacterClasses.WhitespaceCharacters, StringSplitOptions.RemoveEmptyEntries);

                    if (parts.Length < 2) { continue; }

                    var hashes = new ulong[parts.Length];
                    hashes[0] = Data.IgnoreCase ? IgnoreCaseHash64(parts[0].AsSpan()) : Hash64(parts[0].AsSpan());
                    ulong combined = hashes[0];
                    for (int i = 1; i < parts.Length; i++)
                    {
                        hashes[i] = Data.IgnoreCase ? IgnoreCaseHash64(parts[i].AsSpan()) : Hash64(parts[i].AsSpan());
                        combined = HashCombine64(combined, hashes[i]);
                    }

                    Data.Hashes.Add(combined);

                    for (int i = 0; i < hashes.Length; i++)
                    {
                        if (Data.MultiGramHashes.Count <= i)
                        {
                            Data.MultiGramHashes.Add(new HashSet<ulong>());
                        }
                        Data.MultiGramHashes[i].Add(hashes[i]);
                    }
                }
                else
                {
                    var hash = Data.IgnoreCase ? IgnoreCaseHash64(w.AsSpan()) : Hash64(w.AsSpan());
                    Data.Hashes.Add(hash);
                }
            }
        }
    }

    [MessagePack.MessagePackObject(keyAsPropertyName: true)]
    public class Word2SenseTrainingData
    {
        public ConcurrentDictionary<ulong, int> HashCount { get; set; } = new ConcurrentDictionary<ulong, int>();
        public ConcurrentDictionary<ulong, ulong[]> Senses { get; set; } = new ConcurrentDictionary<ulong, ulong[]>();
        public ConcurrentDictionary<ulong, string> Words { get; set; } = new ConcurrentDictionary<ulong, string>();
    }
}