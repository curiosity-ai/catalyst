using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models.Native;
using Mosaik.Core;
using UID;

namespace Catalyst.Models
{
    public class LDAModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public int VocabularyBuckets        { get; set; } = 2_000_000;
        public int NumberOfTopics               { get; set; } = 100;
        public float AlphaSum                   { get; set; } = 100; //Dirichlet prior on document-topic vectors
        public float Beta                       { get; set; } = 0.01f; //Dirichlet prior on vocab-topic vectors
        public int SamplingStepCount            { get; set; } = 4; //Number of Metropolis Hasting step
        public int MaximumNumberOfIterations    { get; set; } = 200;

        public int LikelihoodInterval           { get; set; } = 5; //Compute log likelihood over local dataset on this iteration interval
        public int MaximumTokenCountPerDocument { get; set; } = 512; //The threshold of maximum count of tokens per doc
        public int MinimumTokenCountPerDocument { get; set; } = 1;
        public int NumberOfSummaryTermsPerTopic { get; set; } = 10; //The number of words to summarize the topic
        public int NumberOfBurninIterations     { get; set; } = 10;
        public long MemBlockSize                { get; set; } = 0;
        public long AliasMemBlockSize           { get; set; } = 0;
        public KeyValuePair<int, int>[][] LDA_Data { get; set; }
        public ConcurrentDictionary<int, string> Vocabulary { get; set; } = new ConcurrentDictionary<int, string>();
        public HashSet<uint> StopWords { get; set; }
    }

    public sealed class LDA : StorableObject<LDA, LDAModel>, IDisposable
    {
        private LdaState State;
        public LDA(Language language, int version, string tag) : base(language, version, tag)
        {
        }

        public new static async Task<LDA> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new LDA(language, version, tag);
            await a.LoadDataAsync();
            a.State = new LdaState(a.Data, Environment.ProcessorCount);
            a.State.InitializePretrained(a.Data);
            return a;
        }

        public void Train(IEnumerable<IDocument> documents, int threads, IEnumerable<string> stopwords = null)
        {
            if (Data.NumberOfTopics <=1)
            {
                throw new ArgumentException($"Invalid number of topics ({nameof(Data)}.{nameof(Data.NumberOfTopics)}), must be > 1");
            }

            var stopWords = new HashSet<uint>((stopwords ?? StopWords.Snowball.For(Language)).Select(s => Hash(s.AsSpan())));
            var (count, corpusSize) = InitializeVocabulary(documents, stopWords);
            if ((count == 0) || (corpusSize == 0))
            {
                throw new EmptyCorpusException();
            }

            var state = new LdaState(Data, threads);
            state.AllocateDataMemory(count, corpusSize);

            var vocabulary = new ConcurrentDictionary<int, string>();
            foreach (var doc in documents)
            {
                GetTokensAndFrequencies(doc, vocabulary, stopWords, out var tokenCount, out var tokenIndices, out var tokenFrequencies);

                if (tokenCount >= Data.MinimumTokenCountPerDocument)
                {
                    var docIndex = state.FeedTrain(Data, tokenIndices, tokenCount, tokenFrequencies);
                }

                ArrayPool<int>.Shared.Return(tokenIndices);
                ArrayPool<double>.Shared.Return(tokenFrequencies);
            }
            state.CompleteTrain();

            state.ReadModelFromTrainedLDA(Data);
            Data.Vocabulary = vocabulary;
            Data.StopWords = stopWords;
            State = state;
        }


        public bool TryPredict(IDocument document, out LDATopic[] topics)
        {
            if(State is LdaState state) //Copy the reference
            {
                GetTokensAndFrequencies(document, Data.Vocabulary, Data.StopWords, out var tokenCount, out var tokenIndices, out var tokenFrequencies);
                topics = state.Predict(Data, tokenIndices, tokenCount, tokenFrequencies, false); 
                ArrayPool<int>.Shared.Return(tokenIndices);
                ArrayPool<double>.Shared.Return(tokenFrequencies);

                return true;
            }
            else
            {
                topics = Array.Empty<LDATopic>();
                return false;
            }
        }
        
        public bool TryDescribeTopic(int topicID, out LDATopicDescription topicDescription)
        {
            if (State is LdaState state) //Copy the reference
            {
                var vocabulary = Data.Vocabulary;

                var topic = state.DescribeTopic(topicID);

                var tokens = new Dictionary<string, float>();

                foreach(var kv in topic)
                {
                    tokens[vocabulary[kv.Key]] =  kv.Value;
                }

                topicDescription = new LDATopicDescription(topicID, tokens);

                return true;
            }
            else
            {
                topicDescription = null;
                return false;
            }
        }

        private void GetTokensAndFrequencies(IDocument doc, ConcurrentDictionary<int, string> vocabulary, HashSet<uint> stopWords,  out int tokenCount, out int[] tokenIndices, out double[] tokenFrequencies)
        {
            var tokens = doc.SelectMany(s => s.GetCapturedTokens()).Where(t => ShouldKeepToken(stopWords, t)).Take(Data.MaximumTokenCountPerDocument).ToArray();
            var groups = tokens.Select(tk => TokenToIndex(tk, vocabulary)).GroupBy(i => i).ToArray();
            tokenCount = groups.Length;
            tokenIndices = ArrayPool<int>.Shared.Rent(tokenCount);
            tokenFrequencies = ArrayPool<double>.Shared.Rent(tokenCount);
            for (int i = 0; i < groups.Length; i++)
            {
                tokenIndices[i] = groups[i].Key;
                tokenFrequencies[i] = groups[i].Count();
            }
        }
        
        private (int count, long corpusSize) InitializeVocabulary(IEnumerable<IDocument> documents, HashSet<uint> stopWords)
        {
            int count = 0;
            long corpusSize = 0;
            foreach(var doc in documents)
            {
                var tokenCount = doc.SelectMany(s => s.GetCapturedTokens()).Where(t => ShouldKeepToken(stopWords, t)).Take(Data.MaximumTokenCountPerDocument).Count();

                if(tokenCount >= Data.MinimumTokenCountPerDocument)
                {
                    count++;
                    corpusSize += 2 * tokenCount + 1;
                }
            }
            return (count, corpusSize);
        }

        private static bool ShouldKeepToken(HashSet<uint> stopWords, IToken tk)
        {
            //We keep both NONE when there is no POS tagging on the document (i.e. POS == NONE) or the token represents a merged set of tokens (Tokens always return PartOfSpeech.X (i.e. from entities being captured) - see Tokens.cs

            bool filterPartOfSpeech = !(tk.POS == PartOfSpeech.ADJ || tk.POS == PartOfSpeech.NOUN || tk.POS == PartOfSpeech.PROPN || tk.POS == PartOfSpeech.NONE || tk.POS == PartOfSpeech.X);

            bool skipIfTooSmall = (tk.Length < 3);

            //We ignore skipping non-letter POS = X, as this is due to multiple entity tokens, and we would like to keep entities them for the LDA calculation (and they'll probably have whitespaces besides letters/digits)
            bool skipIfNotAllLetterOrDigit = (tk.POS != PartOfSpeech.X) && !(tk.ValueAsSpan.IsAllLetterOrDigit());

            bool skipIfStopWord = stopWords.Contains(Hash(tk.ValueAsSpan));

            //Heuristic for ordinal numbers (i.e. 1st, 2nd, 33rd, etc)
            bool skipIfMaybeOrdinal = (tk.ValueAsSpan.IndexOfAny(new char[] { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }, 0) >= 0 &&
                                       tk.ValueAsSpan.IndexOfAny(new char[] { 't', 'h', 's', 't', 'r', 'd' }, 0) >= 0 &&
                                       tk.ValueAsSpan.IndexOfAny(new char[] { 'a', 'b', 'c', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z' }, 0) < 0);

            bool skipThisToken = filterPartOfSpeech || skipIfTooSmall || skipIfNotAllLetterOrDigit || skipIfStopWord || skipIfMaybeOrdinal;
            return !skipThisToken;
        }

        private int TokenToIndex(IToken token, ConcurrentDictionary<int, string> vocabulary)
        {
            var index = (int)(Hash(token.ValueAsSpan) % Data.VocabularyBuckets);
            
            if(!vocabulary.ContainsKey(index))
            {
                vocabulary[index] = token.Value; //Only add here to not have to materialize every token as a string
            }
            return index;
        }

        private static uint Hash(ReadOnlySpan<char> word)
        {
            return (uint)word.CaseSensitiveHash32();
        }

        public void Dispose()
        {
            if(State is object)
            {
                State.Dispose();
                State = null;
            }
        }

        public struct LDATopic
        {
            public int TopicID;
            public float Score;

            public LDATopic(int topicID, float score)
            {
                TopicID = topicID;
                Score = score;
            }
        }

        /// <summary>
        /// Provide details about the topics discovered by <a href="https://arxiv.org/abs/1412.1576">LightLDA.</a>
        /// </summary>
        public class LDATopicDescription
        {
            public int TopicID { get; private set; }
            public IReadOnlyDictionary<string, float> Tokens { get; private set; }

            public LDATopicDescription(int topicID, Dictionary<string, float> tokens)
            {
                TopicID = topicID;
                Tokens = tokens;
            }

            public override string ToString()
            {
                var sb = StringExtensions.StringBuilderPool.Rent();
                foreach(var tk in Tokens.OrderByDescending(kv => kv.Value))
                {
                    sb.Append(tk.Key).Append('[').Append(Math.Round(tk.Value, 3)).Append("] ");
                }
                var s = sb.ToString();
                StringExtensions.StringBuilderPool.Return(sb);
                return s;
            }
        }

        private sealed class LdaState : IDisposable
        {
            private readonly object _preparationSyncRoot;
            private readonly object _testSyncRoot;
            private bool _predictionPreparationDone;
            private LdaSingleBox _ldaTrainer;

            private LdaState()
            {
                _preparationSyncRoot = new object();
                _testSyncRoot = new object();
            }

            internal LdaState(LDAModel model, int numberOfThreads) : this()
            {
                _ldaTrainer = new LdaSingleBox(
                                  numTopic          : model.NumberOfTopics,
                                  numVocab          : model.VocabularyBuckets,
                                  alpha             : model.AlphaSum,
                                  beta              : model.Beta,
                                  numIter           : model.MaximumNumberOfIterations,
                                  likelihoodInterval: model.LikelihoodInterval,
                                  numThread         : numberOfThreads,
                                  mhstep            : model.SamplingStepCount,
                                  numSummaryTerms   : model.NumberOfSummaryTermsPerTopic,
                                  denseOutput       : false,
                                  maxDocToken       : model.MaximumTokenCountPerDocument);
            }

            internal void InitializePretrained(LDAModel model)
            {
                _ldaTrainer.AllocateModelMemory(model.VocabularyBuckets, model.NumberOfTopics, model.MemBlockSize, model.AliasMemBlockSize);
                Debug.Assert(model.VocabularyBuckets == model.LDA_Data.Length);

                for (int termID = 0; termID < model.VocabularyBuckets; termID++)
                {
                    var kvs          = model.LDA_Data[termID];
                    var topicId      = kvs.Select(kv => kv.Key).ToArray();
                    var topicProb    = kvs.Select(kv => kv.Value).ToArray();
                    var termTopicNum = topicId.Length;

                    _ldaTrainer.SetModel(termID, topicId, topicProb, termTopicNum);
                }

                //do the preparation
                if (!_predictionPreparationDone)
                {
                    lock (_preparationSyncRoot)
                    {
                        _ldaTrainer.InitializeBeforeTest();
                        _predictionPreparationDone = true;
                    }
                }
            }

            internal void ReadModelFromTrainedLDA(LDAModel model)
            {
                _ldaTrainer.GetModelStat(out var memBlockSize, out var aliasMemBlockSize);
                model.MemBlockSize = memBlockSize;
                model.AliasMemBlockSize = aliasMemBlockSize;
                Debug.Assert(_ldaTrainer.NumVocab == model.VocabularyBuckets);

                model.LDA_Data = Enumerable.Range(0, _ldaTrainer.NumVocab)
                                           .Select(i => _ldaTrainer.GetModel(i))
                                           .ToArray();
            }

            internal void AllocateDataMemory(int docNum, long corpusSize)
            {
                _ldaTrainer.AllocateDataMemory(docNum, corpusSize);
            }

            internal int FeedTrain(LDAModel model, ReadOnlySpan<int> tokenIndices, int tokenCount, ReadOnlySpan<double> frequency)
            {
                if (tokenCount < model.MinimumTokenCountPerDocument)
                {
                    return 0;
                }

                return _ldaTrainer.LoadDoc(tokenIndices, frequency, tokenCount, model.VocabularyBuckets);
            }
            internal void CompleteTrain()
            {
                //allocate all kinds of in memory sample tables
                _ldaTrainer.InitializeBeforeTrain();

                //call native lda trainer to perform the multi-thread training
                _ldaTrainer.Train(""); /* Need to pass in an empty string */
            }

            internal LDATopic[] Predict(LDAModel model, ReadOnlySpan<int> tokenIndices, int tokenCount, ReadOnlySpan<double> frequency, bool reset)
            {
                // Prediction for a single document.
                // LdaSingleBox.InitializeBeforeTest() is NOT thread-safe.
                if (!_predictionPreparationDone)
                {
                    lock (_preparationSyncRoot)
                    {
                        if (!_predictionPreparationDone)
                        {
                            //do some preparation for building tables in native c++
                            _ldaTrainer.InitializeBeforeTest();
                            _predictionPreparationDone = true;
                        }
                    }
                }

                if (tokenCount == 0) return Array.Empty<LDATopic>();

                var retTopics = _ldaTrainer.TestDoc(tokenIndices, frequency, tokenCount, model.NumberOfBurninIterations, reset);
                var normFactor = 1f/retTopics.Sum(kv => kv.Value);
                return retTopics.OrderByDescending(t => t.Value).Select(kv => new LDATopic(kv.Key, kv.Value * normFactor)).ToArray();
            }


            internal KeyValuePair<int, float>[] DescribeTopic(int topicID)
            {
                return _ldaTrainer.GetTopicSummary(topicID);
            }

            public void Dispose()
            {
                _ldaTrainer.Dispose();
            }
        }
    }
}
