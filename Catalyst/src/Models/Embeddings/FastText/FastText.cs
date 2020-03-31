using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using MessagePack;
using Microsoft.Extensions.Logging;
using Mosaik.Core;
using UID;

namespace Catalyst.Models
{
    [FormerName("Mosaik.NLU.Models", "Vectorizer")]
    public partial class FastText : StorableObject<FastText, FastTextData>, ITrainableModel
    {
        #region Constants

        public const char _BOW_ = '<';
        public const char _EOW_ = '>';
        public const string _EOS_ = "</s>";
        public const string _CONTEXTWINDOW_ = "_CONTEXT_WINDOW_";
        public const string _BOW_EOS_EOW_ = "<</s>>";
        public const string _LABEL_ = "_LABEL_";
        private static int _LABEL_HASH_ = Hashes.IgnoreCaseHash32(_LABEL_);
        private static PartOfSpeech[] POS = Enum.GetValues(typeof(PartOfSpeech)).Cast<PartOfSpeech>().ToArray();
        private static readonly uint[] POS_Hashes = Enum.GetValues(typeof(PartOfSpeech)).Cast<PartOfSpeech>().Select(pos => (uint)Hashes.CaseSensitiveHash32(pos.ToString())).ToArray();
        private static readonly uint[] Language_Hashes = Enum.GetValues(typeof(Language)).Cast<Language>().Select(lang => (uint)Hashes.CaseSensitiveHash32(lang.ToString())).ToArray();
        public static uint _HashEOS_; //Initialized on creation, as it depends on POS being already initialized - otherwise might run into a race condition on who's created first
        private const int MAX_VOCAB_SIZE = 30000000;
        private const int NEGATIVE_TABLE_SIZE = 10_000_000;
        private const string __DATA_FILE__ = "data";

        #endregion Constants

        public int[] EntrySubwordsBegin;
        public int[] EntrySubwordsLength;
        public int[] EntrySubwordsFlatten;

        //public int[][] EntrySubwords;
        public int[] EntryDiscardProbability;

        public long NumberOfTokens = 0;

        //Variables used during training
        private long TokenCount = 0;

        private long PartialTokenCount = 0;
        private int[] NegativeTable;
        private Barrier TrainingBarrier; //Barrier to force all threads to be in sync between Epochs

        public IMatrix Wi;
        public IMatrix Wo;

        public int GradientLength;
        public int HiddenLength;
        public int OutputLength;

        private List<List<int>> HS_Paths = new List<List<int>>();
        private List<List<bool>> HS_Codes = new List<List<bool>>();
        private List<HSNode> HS_Tree = new List<HSNode>();

        private ObjectPool<ThreadState> ThreadStatePool;

        public TrainingHistory TrainingHistory => Data.TrainingHistory;

        public event EventHandler<TrainingUpdate> TrainingStatus;

        public FastText(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
            //Has to be done here, after POSHashes is initialized
            _HashEOS_ = HashToken(_EOS_.AsSpan(), PartOfSpeech.NONE, Language.Any);
        }

        public VectorizerTrainingData GetPreviousData(int version)
        {
            return DataStore.GetData<VectorizerTrainingData>(Language, nameof(VectorizerTrainingData), version, Tag + "-" + __DATA_FILE__, compress: false);
        }

        public override async Task StoreAsync()
        {
            var wiStream = await DataStore.OpenWriteAsync(Language, nameof(FastTextData) + "-Matrix", Version, Tag + "-wi");
            var woStream = await DataStore.OpenWriteAsync(Language, nameof(FastTextData) + "-Matrix", Version, Tag + "-wo");

            wiStream.SetLength(0);
            woStream.SetLength(0);

            Wi.ToStream(wiStream, Data.VectorQuantization);
            Wo.ToStream(woStream, Data.VectorQuantization);

            wiStream.Close();
            woStream.Close();

            //Data.TranslateTable = TranslateTable;
            await base.StoreAsync();
        }

        public new static async Task<bool> DeleteAsync(Language language, int version, string tag)
        {
            var a = new FastText(language, version, tag);
            bool deleted = false;
            deleted |= await DataStore.DeleteAsync(language, nameof(FastTextData) + "-Matrix", version, tag + "-wi");
            deleted |= await DataStore.DeleteAsync(language, nameof(FastTextData) + "-Matrix", version, tag + "-wo");

            if (ObjectStore.TryGetFormerNames(nameof(FastTextData), out var formerNames))
            {
                foreach (var fn in formerNames)
                {
                    deleted |= await DataStore.DeleteAsync(language, fn + "Matrix", version, tag + "-wi"); //On purpose without the '-', old bug
                    deleted |= await DataStore.DeleteAsync(language, fn + "Matrix", version, tag + "-wo"); //On purpose without the '-', old bug
                }
            }

            deleted |= await a.DeleteDataAsync();
            return deleted;
        }

        public bool TryGetTrainingData(out VectorizerTrainingData previousTrainingCorpus)
        {
            previousTrainingCorpus = null;
            if (!Data.IsTrained) { return false; }
            if (!Data.StoreTrainingData) { return false; }

            try
            {
                previousTrainingCorpus = DataStore.GetData<VectorizerTrainingData>(Language, nameof(VectorizerTrainingData), Version, Tag + "-" + __DATA_FILE__, false);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public new static async Task<FastText> FromStoreAsync(Language language, int version, string tag)
        {
            return await FromStoreAsync_Internal(language, version, tag);
        }

        public static async Task<FastText> FromStoreAsync_Internal(Language language, int version, string tag)
        {
            var a = new FastText(language, version, tag);
            await a.LoadDataAsync();

            (Stream wiStream, Stream woStream) = await a.GetMatrixStreamsAsync();

            a.Wi = Matrix.FromStream(wiStream, a.Data.VectorQuantization);
            a.Wo = Matrix.FromStream(woStream, a.Data.VectorQuantization);
            wiStream.Close();
            woStream.Close();

            //a.TranslateTable = a.Data.TranslateTable;
            a.GradientLength = a.Data.Dimensions;
            a.HiddenLength = a.Data.Dimensions;
            if (a.Data.Type == ModelType.Supervised)
            {
                a.OutputLength = a.Data.LabelCount;
            }
            else
            {
                a.OutputLength = a.Data.EntryCount;
            }
            a.InitializeEntries();
            return a;
        }

        private async Task<(LockedStream wi, LockedStream wo)> GetMatrixStreamsAsync()
        {
            var names = new List<string>(new[] { nameof(FastTextData) + "-Matrix" });

            if (ObjectStore.TryGetFormerNames(nameof(FastTextData), out var formerNames))
            {
                names.AddRange(formerNames.Select(fn => fn + "Matrix")); //On purpose without the '-', old bug
            }

            foreach (var name in names)
            {
                try
                {
                    var wi = await DataStore.OpenReadAsync(Language, name, Version, Tag + "-wi");
                    var wo = await DataStore.OpenReadAsync(Language, name, Version, Tag + "-wo");
                    return (wi, wo);
                }
                catch (FileNotFoundException)
                {
                    //do nothing, will try a different name
                }
            }
            throw new FileNotFoundException(nameof(FastTextData) + "-Matrix");
        }

        public void Train(IEnumerable<IDocument> documents, Func<IToken, bool> ignorePattern = null, ParallelOptions parallelOptions = default, VectorizerTrainingData previousTrainingCorpus = null)
        {
            InputData inputData;
            CancellationToken cancellationToken = parallelOptions?.CancellationToken ?? default;
            using (var scope = Logger.BeginScope($"Training Vectorizer '{Tag}' of type {Data.Type} from documents"))
            {
                if (!(previousTrainingCorpus is null))
                {
                    Logger.LogInformation("Reusing previous training corpus with {COUNT} entries", previousTrainingCorpus.Lines.Count);
                }

                using (var m = new Measure(Logger, "Document parsing", 1))
                {
                    inputData = ProcessDocuments(documents, ignorePattern, parallelOptions);
                    m.SetOperations(inputData.docCount);
                }

                using (var m = new Measure(Logger, "Training vector model " + (Vector.IsHardwareAccelerated ? "using hardware acceleration [" + Vector<float>.Count + "]" : "without hardware acceleration"), inputData.docCount))
                {
                    DoTraining(inputData, cancellationToken, previousTrainingCorpus);
                }
            }
        }

        public void DoTraining(InputData ID, CancellationToken cancellationToken, VectorizerTrainingData previousTrainingCorpus = null)
        {
            // If there are no documents to process then we can't perform any training (maybe documents WERE provided to the Train method but they all had to be ignored for one reason or another - eg. wrong language)
            if (ID.docCount < 1)
                return;

            cancellationToken.ThrowIfCancellationRequested();

            ThreadState[] modelPrivateState;

            using (var m = new Measure(Logger, "Initializing private thread states for training", 1))
            {
                modelPrivateState = Initialize(ID, cancellationToken, previousTrainingCorpus);
            }

            //Create a barrier to force all threads to be in sync between Epochs
            TrainingBarrier = new Barrier(participantCount: modelPrivateState.Length);

            var trainingHistory = modelPrivateState[0].TrainingHistory;
            Debug.Assert(trainingHistory is object);

            using (var m = new Measure(Logger, "Training", Data.Epoch))
            {
                var threads = modelPrivateState.Select(mps =>
                                                            {
                                                                var t = new Thread(() => ThreadTrain(mps));
                                                                t.Priority = Data.ThreadPriority;
                                                                t.Start();
                                                                return t;
                                                            }).ToArray();
                foreach (var t in threads) { t.Join(); }
            }

            cancellationToken.ThrowIfCancellationRequested(); //If the training was canceled, all threads will return, so we throw here

            //Quantize the final matrices if necessary
            if (Data.VectorQuantization != QuantizationType.None)
            {
                using (var m = new Measure(Logger, "Quantizing results", Wi.Rows + Wo.Rows))
                {
                    for (int r = 0; r < Wi.Rows; r++) { Quantize(Wi.GetRow(r)); }
                    for (int r = 0; r < Wo.Rows; r++) { Quantize(Wo.GetRow(r)); }
                }
            }

            Data.TrainingHistory = trainingHistory;
            Data.IsTrained = true;
        }

        public IEnumerable<TokenVector> GetVectors()
        {
            for (int i = 0; i < Data.EntryCount; i++)
            {
                var e = Data.Entries[i];
                if (e.Type == EntryType.Word)
                {
                    yield return new TokenVector(e.Word, GetVector(i), i, e.POS, e.Language, (float)e.Count / (float)NumberOfTokens);
                }
            }
        }

        public IEnumerable<TokenVector> GetDocumentVectors()
        {
            float oneDocFrequency = 1f / Data.LabelCount;
            for (int i = 0; i < Data.EntryCount; i++)
            {
                var e = Data.Entries[i];
                if (e.Type == EntryType.Label)
                {
                    yield return new TokenVector(e.Word, GetVector(i), i, e.POS, e.Language, oneDocFrequency);
                }
            }
        }

        public TokenVector GetTokenVector(int index)
        {
            var e = Data.Entries[index];
            return new TokenVector(e.Word, GetVector(index), index, e.POS, e.Language, (float)e.Count / (float)NumberOfTokens);
        }

        public float[] CompareDocuments(IDocument source, IDocument target)
        {
            //TODO CosineSimilarity is normalizing all the time
            if (TryGetDocumentVector(source, out var sourceVect) && TryGetDocumentVector(target, out var targetVect))
            {
                var baseScore = sourceVect.CosineSimilarityWith(targetVect);

                var wrappedDoc = new DocumentWrapper(target);
                var spanDelta = new float[target.SpansCount];
                for (int i = 0; i < target.SpansCount; i++)
                {
                    wrappedDoc.SetInvisibleSpan(i);
                    if (TryGetDocumentVector(wrappedDoc, out var partialVect))
                    {
                        var partialScore = sourceVect.CosineSimilarityWith(partialVect);
                        spanDelta[i] = baseScore - partialScore;
                    }
                }
                return spanDelta;
            }
            return Array.Empty<float>();
        }

        public bool TryGetDocumentVector(IDocument doc, out float[] vector)
        {
            //TODO: make this thread safe by having more than one MPS that can be used at the same time

            if (Data.Type != ModelType.PVDM) { throw new Exception("GetDocumentVector can only be used on PVDM models"); }


            if (doc.UID.IsNotNull() && TryGetLabelVector(doc.UID.ToString(), out vector))
            {
                return true;
            }

            if (doc.Language != Language)
            {
                vector = default;
                return false;
                /*throw new Exception($"Document language '{doc.Language}'does not match model language '{Language}'");*/
            }

            var tokens = doc.SelectMany(span => span.GetCapturedTokens()).ToArray();

            var tokenHashes = new List<uint>(tokens.Length);

            foreach (var tk in tokens)
            {
                uint hash = HashToken(tk, Language);
                tokenHashes.Add(hash);
            }

            tokenHashes.Add(_HashEOS_);

            var averageVector = new float[Data.Dimensions];

            int tries = 1;
            var tokenIndexes = tokenHashes.Where(hash => Data.EntryHashToIndex.ContainsKey(hash)).Select(hash => Data.EntryHashToIndex[hash]).ToArray();
            var line = new Line(tokenIndexes, new int[0]);

            var predictedVector = new float[Data.Dimensions];

            if (tokenIndexes.Length < Data.ContextWindow * 2)
            {
                vector = default;
                return false;
            }
            var mps = ThreadStatePool.Rent();

            for (int i = 0; i < tries; i++)
            {
                float a = 1.0f / Data.Dimensions;

                for (int j = 0; j < Data.Dimensions; j++)
                {
                    predictedVector[j] = (float)(ThreadSafeFastRandom.NextDouble()) * (2 * a) - a;
                }

                mps.NumberOfExamples = 0;
                mps.Loss = 0f;

                for (int epoch = 0; epoch < Data.Epoch; epoch++)
                {
                    float lr = Data.LearningRate * (1.0f - (float)epoch / (Data.Epoch));
                    PredictPVDM(predictedVector, mps, ref line, lr);
                }

                SIMD.Add(averageVector, predictedVector);
            }
            SIMD.Multiply(averageVector, 1f / tries);
            vector = averageVector;

            ThreadStatePool.Return(mps);

            return true;
        }

        public bool TryGetLabelVector(string label, out float[] vector)
        {
            var hash = HashLabel(label);
            if (Data.EntryHashToIndex.TryGetValue(hash, out int index))
            {
                vector = GetVector(index);
                return true;
            }
            vector = default;
            return false;
        }

        public Span<int> GetEntrySubwords(int index)
        {
            return EntrySubwordsFlatten.AsSpan().Slice(EntrySubwordsBegin[index], EntrySubwordsLength[index]);
        }

        public float[] GetVector(int index)
        {
            var vec = new float[Data.Dimensions];
            var ngrams = GetEntrySubwords(index);
            if (ngrams.Length > 0)
            {
                foreach (var ngram in ngrams) { SIMD.Add(vec, Wi.GetRow(ngram)); }
                SIMD.Multiply(vec, 1f / ngrams.Length);
            }
            return vec;
        }

        public float[] GetVector(string word, Language language, PartOfSpeech pos)
        {
            var ix = GetWordIndex(word, language, pos);

            if (ix >= 0) { return GetVector(ix); }

            var vec = new float[Data.Dimensions];

            var subwords = GetCharNgrams(word);
            subwords = TranslateNgramHashesToIndexes(subwords, language, create: false);
            if (subwords.Count > 0)
            {
                foreach (var ngram in subwords)
                {
                    SIMD.Add(vec, Wi.GetRow((int)ngram));
                }
                SIMD.Multiply(vec, 1f / subwords.Count);
            }
            return vec;
        }

        public (float[] vector, long count) GetVectorAndCount(string word, Language language, PartOfSpeech pos)
        {
            var ix = GetWordIndex(word, language, pos);

            if (ix >= 0) { return (GetVector(ix), Data.Entries[ix].Count); }

            var vec = new float[Data.Dimensions];

            var subwords = GetCharNgrams(word);
            subwords = TranslateNgramHashesToIndexes(subwords, language, create: false);
            if (subwords.Count > 0)
            {
                foreach (var ngram in subwords)
                {
                    SIMD.Add(vec, Wi.GetRow((int)ngram));
                }
                SIMD.Multiply(vec, 1f / subwords.Count);
            }
            return (vec, 0);
        }

        public float[] GetVector(string word, Language language)
        {
            return GetVector(word, Language, GetMostProbablePOSforWord(word));
        }

        public (float[] vector, long count) GetVectorAndCount(string word, Language language)
        {
            return GetVectorAndCount(word, Language, GetMostProbablePOSforWord(word));
        }

        public PartOfSpeech GetMostProbablePOSforWord(string word)
        {
            var candidateHashes = POS.Select(pos => HashToken(word.AsSpan(), pos, Language)).ToArray();
            return candidateHashes.Where(ch => Data.EntryHashToIndex.ContainsKey(ch))
                                  .Select(ch => Data.Entries[Data.EntryHashToIndex[ch]])
                                  .OrderByDescending(e => e.Count)
                                  .FirstOrDefault().POS;
        }

        public int GetWordIndex(string word)
        {
            var pos = GetMostProbablePOSforWord(word);
            var hash = HashToken(new SingleToken(word, pos, 0, 0, Language), Language);
            return Data.EntryHashToIndex.ContainsKey(hash) ? Data.EntryHashToIndex[hash] : -1;
        }

        public int GetWordIndex(string word, Language language, PartOfSpeech pos)
        {
            var hash = HashToken(new SingleToken(word, pos, 0, 0, language), language);
            return Data.EntryHashToIndex.ContainsKey(hash) ? Data.EntryHashToIndex[hash] : -1;
        }

        public float[] GetVector(IToken token, Language language)
        {
            var vec = new float[Data.Dimensions];
            var subwords = GetCharNgrams(token.Value);
            subwords = TranslateNgramHashesToIndexes(subwords, language, create: false);

            subwords.Add((uint)Data.EntryCount + (POS_Hashes[(int)token.POS] % Data.Buckets));

            var hash = HashToken(token, language);
            if (Data.EntryHashToIndex.TryGetValue(hash, out int index))
            {
                subwords.Add((uint)index);
            }

            if (subwords.Count > 0)
            {
                foreach (var ngram in subwords)
                {
                    SIMD.Add(vec, Wi.GetRow((int)ngram));
                }

                SIMD.Multiply(vec, 1f / subwords.Count);
            }
            return vec;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETCOREAPP3_0
        public void Quantize(Span<float> vector)
#else
        public void Quantize(float[] vector)
#endif
        {
            switch (Data.VectorQuantization)
            {
                case QuantizationType.None: return;
                case QuantizationType.OneBit:
                {
                    SIMD.Quantize1Bit(vector);
                    return;
                }
                case QuantizationType.TwoBits:
                {
                    SIMD.Quantize2Bits(vector);
                    return;
                }
                case QuantizationType.FourBits:
                {
                    for (int i = 0; i < vector.Length; i++)
                    {
                        var v = vector[i];
                        if (v < -0.75) { vector[i] = -0.75f; }
                        else if (v < -0.65) { vector[i] = -0.65f; }
                        else if (v < -0.55) { vector[i] = -0.55f; }
                        else if (v < -0.45) { vector[i] = -0.45f; }
                        else if (v < -0.35) { vector[i] = -0.35f; }
                        else if (v < -0.25) { vector[i] = -0.25f; }
                        else if (v < -0.15) { vector[i] = -0.15f; }
                        else if (v < -0.05) { vector[i] = -0.05f; }
                        else if (v < 0.05) { vector[i] = 0.05f; }
                        else if (v < 0.15) { vector[i] = 0.15f; }
                        else if (v < 0.25) { vector[i] = 0.25f; }
                        else if (v < 0.35) { vector[i] = 0.35f; }
                        else if (v < 0.45) { vector[i] = 0.45f; }
                        else if (v < 0.55) { vector[i] = 0.55f; }
                        else if (v < 0.65) { vector[i] = 0.65f; }
                        else { vector[i] = 0.75f; }
                    } //                                -1,1 -> 0,2 -> 0,16 -> 0f,2f, -> -1f,1f
                    return;
                }
            }
        }

        public (string label, float score) PredictMax(IDocument doc, int maxTokens = -1)
        {
            if (Language != Language.Any && doc.Language != Language) { throw new Exception($"Document language ({doc.Language}) not the same as model language ({Language})"); }
            if (Data.Type != ModelType.Supervised) { throw new Exception("Predict can only be called on Supervised models"); }

            var state = ThreadStatePool.Rent();
            
            IEnumerable<IToken> tokens;
            if (maxTokens <= 0)
            {
                tokens = doc.SelectMany(span => span.GetCapturedTokens());
                maxTokens = doc.TokensCount;
            }
            else
            {
                tokens = doc.SelectMany(span => span.GetCapturedTokens()).Take(maxTokens);
                maxTokens = Math.Min(maxTokens, doc.TokensCount);
            }

            var tokenHashes = new List<uint>(maxTokens);
            var tokenNGramsIndexes = new List<int>(maxTokens);
            foreach (var tk in tokens)
            {
                if (tk.Value.Length > 0)
                {
                    uint hash = HashToken(tk, Language);
                    tokenHashes.Add(hash);
                    var subwords = GetCharNgrams(tk.Value);
                    subwords = TranslateNgramHashesToIndexes(subwords, Language, create: false);
                    if (subwords.Any())
                    {
                        tokenNGramsIndexes.AddRange(subwords.Select(h => (int)h));
                    }
                }
            }

            var entries = tokenHashes.Where(hash => Data.EntryHashToIndex.ContainsKey(hash))
                                     .Select(hash => Data.EntryHashToIndex[hash]).ToArray();

            var wordNgrams = GetWordNGrams(entries, create: false);
            entries = entries.Concat(wordNgrams).Concat(tokenNGramsIndexes).ToArray();

            if (entries.Length > 0)
            {
                ComputeHidden(state, entries);

                switch (Data.Loss)
                {
                    case LossType.SoftMax            : ComputeOutputSoftmax(state);        break;
                    case LossType.NegativeSampling   : ComputeOutputBinaryLogistic(state); break;
                    case LossType.HierarchicalSoftMax: ComputeOutputBinaryLogistic(state); break;
                    case LossType.OneVsAll           : ComputeOutputBinaryLogistic(state); break;
                }

                var index = state.Output.Argmax();
                var result = (Data.Labels[index].Word, state.Output[index]);
                ThreadStatePool.Return(state);
                return result;
            }
            else
            {
                return (null, float.NaN);
            }
        }

        public Dictionary<string, float> Predict(IDocument doc)
        {
            if (Language != Language.Any && doc.Language != Language) { throw new Exception($"Document language ({doc.Language}) not the same as model language ({Language})"); }
            if (Data.Type != ModelType.Supervised) { throw new Exception("Predict can only be called on Supervised models"); }

            var state = ThreadStatePool.Rent();
            var tokens = doc.SelectMany(span => span.GetCapturedTokens()).ToArray();

            var tokenHashes = new List<uint>(tokens.Length);
            var tokenNGramsIndexes = new List<int>(tokens.Length);
            foreach (var tk in tokens)
            {
                uint hash = HashToken(tk, Language);
                tokenHashes.Add(hash);
                var subwords = GetCharNgrams(tk.Value);
                subwords = TranslateNgramHashesToIndexes(subwords, Language, create: false);
                if (subwords.Any())
                {
                    tokenNGramsIndexes.AddRange(subwords.Select(h => (int)h));
                }
            }

            var entries = tokenHashes.Where(hash => Data.EntryHashToIndex.ContainsKey(hash))
                                     .Select(hash => Data.EntryHashToIndex[hash]).ToArray();

            var wordNgrams = GetWordNGrams(entries, create: false);
            entries = entries.Concat(wordNgrams).Concat(tokenNGramsIndexes).ToArray();

            if (entries.Length > 0)
            {
                ComputeHidden(state, entries);
                switch(Data.Loss)
                {
                    case LossType.SoftMax:             ComputeOutputSoftmax(state);        break;
                    case LossType.NegativeSampling:    ComputeOutputBinaryLogistic(state); break;
                    case LossType.HierarchicalSoftMax: ComputeOutputBinaryLogistic(state); break;
                    case LossType.OneVsAll:            ComputeOutputBinaryLogistic(state); break;
                }
            }

            var ans = new Dictionary<string, float>(OutputLength);

            for (int i = 0; i < state.Output.Length; i++)
            {
                ans[Data.Labels[i].Word] = state.Output[i];
            }

            ThreadStatePool.Return(state);

            return ans;
        }

        private void ThreadTrain(ThreadState state)
        {
            long localTokenCount = 0;
            var sinceEpochWatch = Stopwatch.StartNew();
            var sinceBeginingWatch = Stopwatch.StartNew();

            float progress = 0f, lr = Data.LearningRate;

            float baseLR = Data.LearningRate / 200;

            float nextProgressReport = 0f;
            for (int epoch = 0; epoch < Data.Epoch; epoch++)
            {

                for (int i = 0; i < state.Corpus.Length; i++)
                {
                    localTokenCount += state.Corpus[i].EntryIndexes.Length;

                    switch (Data.Type)
                    {
                        case ModelType.CBow:       { CBow(state, ref state.Corpus[i].EntryIndexes, lr);     break; }
                        case ModelType.Skipgram:   { Skipgram(state, ref state.Corpus[i].EntryIndexes, lr); break; }
                        case ModelType.Supervised: { Supervised(state, ref state.Corpus[i], lr);            break; }
                        case ModelType.PVDM:       { PVDM(state, ref state.Corpus[i], lr);                  break; }
                        case ModelType.PVDBow:     { PVDBow(state, ref state.Corpus[i], lr);                break; }
                    }

                    if (localTokenCount > Data.LearningRateUpdateRate)
                    {
                        if (state.CancellationToken.IsCancellationRequested) { return; } //Cancelled the training, so return from the thread

                        progress = (float)(TokenCount) / (Data.Epoch * NumberOfTokens);

                        var x10 = (float)(TokenCount) / (10 * NumberOfTokens);

                        //lr = Data.LearningRate * (1.0f - progress);
                        //plot abs(cos(x))*0.98^x from x = [0,100]
                        //lr = (float)(baseLR + (Data.LearningRate - baseLR) * (0.5 + 0.5 * Math.Sin(100 * progress))); //Cyclic loss rate
                        lr = (float)(baseLR + (Data.LearningRate - baseLR) * Math.Abs(Math.Cos(200 * x10)) * Math.Pow(0.98, 100 * x10)); //Cyclic loss rate, scaled for 10 epoch

                        Interlocked.Add(ref TokenCount, localTokenCount);
                        Interlocked.Add(ref PartialTokenCount, localTokenCount);

                        localTokenCount = 0;

                        if (state.ThreadID == 0 && progress > nextProgressReport)
                        {

                            nextProgressReport += 0.01f; //Report every 1%
                            var loss = state.GetLoss();
                            var ws = (double)(Interlocked.Exchange(ref PartialTokenCount, 0)) / sinceEpochWatch.Elapsed.TotalSeconds;
                            sinceEpochWatch.Restart();
                            float wst = (float)(ws / Data.Threads);

                            Logger.LogInformation("At {PROGRESS:n1}%, w/s/t: {WST:n0}, w/s: {WS:n0}, loss at epoch {EPOCH}/{MAXEPOCH}: {LOSS:n5}", (progress * 100), wst, ws, epoch + 1, Data.Epoch, loss);

                            var update = new TrainingUpdate().At(progress, Data.Epoch, loss)
                                                             .Processed(Interlocked.Read(ref TokenCount), sinceBeginingWatch.Elapsed);
                            state.TrainingHistory.Append(update);

                            TrainingStatus?.Invoke(this, update);
                        }
                    }
                }
                TrainingBarrier.SignalAndWait();
            }
            if (state.ThreadID == 0)
            {
                state.TrainingHistory.ElapsedTime = sinceBeginingWatch.Elapsed;
            }
        }

        private bool ShouldDiscard(ThreadState mps, int index)
        {
            //if(Type == VectorModelType.Supervised) { return false; } //Unnecessary as we don't call this function during supervised traning - differently than the original FastText
            return (ThreadSafeFastRandom.Next() < EntryDiscardProbability[index]);
        }

        private void CBow(ThreadState mps, ref int[] l, float lr)
        {
            if (l is null) { return; }
            var bow = new List<int>();
            int len = l.Length;
            int cw = Data.ContextWindow;
            bool useWNG = Data.CBowUseWordNgrams;
            if (len < cw * 2) { return; }
            for (int w = 0; w < len; w++)
            {
                if (!ShouldDiscard(mps, l[w])) // Not exactly the same as the original FastText (they discard when creating a line from the text on every iteration), but as we pre-process the text, this is the best place to do it without allocating anything extra
                {
                    int contextSize = ThreadSafeFastRandom.Next(1, cw);
                    bow.Clear();
                    for (int c = -contextSize; c <= contextSize; c++)
                    {
                        if (c != 0 && w + c >= 0 && w + c < len)
                        {
                            var ngrams = GetEntrySubwords(l[w + c]);
                            foreach (var ng in ngrams)
                            {
                                bow.Add(ng);
                            }
                        }
                    }

                    if (useWNG)
                    {
                        if(w > contextSize)
                        {
                            var wng_before = GetWordNGrams(l.AsSpan().Slice(w - contextSize, contextSize), false);
                            bow.AddRange(wng_before);
                        }
                        if(w < len - contextSize)
                        {
                            var wng_after = GetWordNGrams(l.AsSpan().Slice(w + 1, contextSize), false);
                            bow.AddRange(wng_after);
                        }
                    }

                    var bow_a = bow.ToArray();
                    Update(mps, bow_a, l[w], lr);
                }
            }
        }

        private void Skipgram(ThreadState mps, ref int[] l, float lr)
        {
            if (l is null) { return; }
            int len = l.Length;
            int cw = Data.ContextWindow;
            if (len < cw * 2) { return; }
            for (int w = 0; w < len; w++)
            {
                if (!ShouldDiscard(mps, l[w])) // Not exactly the same as the original FastText (they discard when creating a line from the text on every iteration), but as we pre-process the text, this is the best place to do it without allocating anything extra
                {
                    int contextSize = ThreadSafeFastRandom.Next(1, cw);
                    var ngrams = GetEntrySubwords(l[w]);
                    for (int c = -contextSize; c <= contextSize; c++)
                    {
                        if (c != 0 && w + c >= 0 && w + c < len)
                        {
                            Update(mps, ngrams, l[w + c], lr);
                        }
                    }
                }
            }
        }

        private void Supervised(ThreadState mps, ref Line l, float lr)
        {
            if (l.Labels.Length == 0 || l.EntryIndexes.Length == 0) { return; }
            if(Data.Loss == LossType.OneVsAll)
            {
                UpdateOneVsAll(mps, l.EntryIndexes, l.Labels, lr);
            }
            else
            {
                int r = l.Labels.Length == 1 ? 0 : ThreadSafeFastRandom.Next(0, l.Labels.Length - 1);
                Update(mps, l.EntryIndexes, l.Labels[r], lr);
            }
        }

        private void PVDBow(ThreadState mps, ref Line l, float lr)
        {
            int len = l.EntryIndexes.Length;
            if (len == 0) { return; }
            for (int w = 0; w < len; w++)
            {
                if (!ShouldDiscard(mps, l.EntryIndexes[w])) // Not exactly the same as the original FastText (they discard when creating a line from the text on every iteration), but as we pre-process the text, this is the best place to do it without allocating anything extra
                {
                    int contextSize = ThreadSafeFastRandom.Next(1, Data.ContextWindow);
                    for (int c = -contextSize; c <= contextSize; c++)
                    {
                        if (c != 0 && w + c >= 0 && w + c < len)
                        {
                            Update(mps, l.Labels, l.EntryIndexes[w + c], lr);
                        }
                    }
                }
            }
        }

        //source: https://arxiv.org/pdf/1405.4053v2.pdf
        private void PVDM(ThreadState mps, ref Line l, float lr)
        {
            int cw = Data.ContextWindow;
            int len = l.EntryIndexes.Length - cw;
            if (len <= 0) { return; }
            var bow = new List<int>(len);

            for (int w = cw; w < len; w++)
            {
                bow.Clear();
                for (int c = -cw; c <= cw; c++)
                {
                    //if(c != 0) - Can't skip words if we are doing word n-grams after, otherwise word n-grams are wrong
                    {
                        bow.Add(l.EntryIndexes[w + c]);
                    }
                }

                AppendWordNGrams(bow, create: false);

                bow.AddRange(l.Labels);

                var bow_a = bow.ToArray();
                Update(mps, bow_a, l.EntryIndexes[w], lr);
            }
        }
#if NETCOREAPP3_0
        private void PredictPVDM(Span<float> predictionVector, ThreadState mps, ref Line l, float lr)
#else
        private void PredictPVDM(float[] predictionVector, ThreadState mps, ref Line l, float lr)
#endif
        {
            int cw = Data.ContextWindow;
            int len = l.EntryIndexes.Length - cw;
            var bow = new List<int>(cw * 2);
            for (int w = cw; w < len; w++)
            {
                bow.Clear();
                for (int c = -cw; c <= cw; c++)
                {
                    //if (c != 0) - Can't skip words if we are doing word n-grams after, otherwise word n-grams are wrong
                    {
                        bow.Add(l.EntryIndexes[w + c]);
                    }
                }

                AppendWordNGrams(bow, create: false);

                var bow_a = bow.ToArray();
                UpdatePredictionOnly(predictionVector, mps, bow_a, l.EntryIndexes[w], lr);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Update(ThreadState state, Span<int> input, int target, float lr)
        {
            if (input.Length == 0) { return; }
            state.NumberOfExamples++;
            ComputeHidden(state, input);
            switch (Data.Loss)
            {
                case LossType.NegativeSampling: { state.Loss += ComputeNegativeSampling(state, target, lr); break; }
                case LossType.HierarchicalSoftMax: { state.Loss += ComputeHierarchicalSoftMax(state, target, lr); break; }
                case LossType.SoftMax: { state.Loss += ComputeSoftmax(state, target, lr); break; }
            }

            if (Data.Type == ModelType.Supervised || Data.Type == ModelType.PVDM)
            {
                SIMD.Multiply(state.Gradient, 1.0f / input.Length);
            }

            foreach (var ix in input)
            {
                Wi.AddToRow(state.Gradient, ix);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateOneVsAll(ThreadState state, Span<int> input, int[] targets, float lr)
        {
            //Only for supervised models
            Debug.Assert(Data.Type == ModelType.Supervised && Data.Loss == LossType.OneVsAll);
            if (input.Length == 0) { return; }
            state.NumberOfExamples += targets.Length;
            ComputeHidden(state, input);

            state.Loss += ComputeOneVsAllLoss(state, targets, lr);

            SIMD.Multiply(state.Gradient, 1.0f / input.Length);

            foreach (var ix in input)
            {
                Wi.AddToRow(state.Gradient, ix);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETCOREAPP3_0
        private void UpdatePredictionOnly(Span<float> predictionVector, ThreadState state, Span<int> input, int target, float lr)
#else
        private void UpdatePredictionOnly(float[] predictionVector, ThreadState state, Span<int> input, int target, float lr)
#endif
        {
            if (input.Length == 0) { return; }
            state.NumberOfExamples++;

            ComputeHiddenForPrediction(state.Hidden, input, predictionVector);

            switch (Data.Loss)
            {
                case LossType.NegativeSampling:    { state.Loss += ComputeNegativeSampling(state, target, lr, addToOutput: false);    break; }
                case LossType.HierarchicalSoftMax: { state.Loss += ComputeHierarchicalSoftMax(state, target, lr, addToOutput: false); break; }
                case LossType.SoftMax:             { state.Loss += ComputeSoftmax(state, target, lr, addToOutput: false);             break; }
            }

            SIMD.Multiply(state.Gradient, 1.0f / (input.Length + 1));

            SIMD.Add(predictionVector, state.Gradient);
        }

        private void SetTargetCounts(List<long> counts)
        {
            Debug.Assert(counts.Count == OutputLength);

            if (Data.Loss == LossType.NegativeSampling)
            {
                InitializeTableNegatives(counts);
            }
            if (Data.Loss == LossType.HierarchicalSoftMax)
            {
                InitializeTree(counts);
            }
        }

        private void InitializeTableNegatives(List<long> counts)
        {
            if (counts.Count < 2)
            {
                // If we don't perform this check and there is only a single label specified then the GetNegative method will get stuck in an infinite loop
                throw new Exception("It will not be possible to use NegativeSampling without specifying multiple labels");
            }

            double z = 0.0;
            var negatives = new List<int>();
            for (int i = 0; i < counts.Count; i++)
            {
                z += Math.Pow(counts[i], 0.5);
            }
            for (int i = 0; i < counts.Count; i++)
            {
                double c = Math.Pow(counts[i], 0.5);
                for (int j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++)
                {
                    negatives.Add(i);
                }
            }
            negatives.Shuffle();
            NegativeTable = negatives.ToArray();
        }

        private void InitializeTree(List<long> counts)
        {
            HS_Tree.Capacity = 2 * OutputLength - 1;

            for (int i = 0; i < 2 * OutputLength - 1; i++)
            {
                HS_Tree.Add(new HSNode() { parent = -1, left = -1, right = -1, count = (long)1e15, binary = false });
            }

            for (int i = 0; i < OutputLength; i++)
            {
                HS_Tree[i].count = counts[i];
            }

            int leaf = OutputLength - 1;
            int node = OutputLength;
            for (int i = OutputLength; i < 2 * OutputLength - 1; i++)
            {
                var mini = new int[2];
                for (int j = 0; j < 2; j++)
                {
                    if (leaf >= 0 && HS_Tree[leaf].count < HS_Tree[node].count)
                    {
                        mini[j] = leaf--;
                    }
                    else
                    {
                        mini[j] = node++;
                    }
                }
                HS_Tree[i].left = mini[0];
                HS_Tree[i].right = mini[1];
                HS_Tree[i].count = HS_Tree[mini[0]].count + HS_Tree[mini[1]].count;
                HS_Tree[mini[0]].parent = i;
                HS_Tree[mini[1]].parent = i;
                HS_Tree[mini[1]].binary = true;
            }

            for (int i = 0; i < OutputLength; i++)
            {
                var path = new List<int>();
                var code = new List<bool>();
                int j = i;
                while (HS_Tree[j].parent != -1)
                {
                    path.Add(HS_Tree[j].parent - OutputLength);
                    code.Add(HS_Tree[j].binary);
                    j = HS_Tree[j].parent;
                }
                HS_Paths.Add(path);
                HS_Codes.Add(code);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeNegativeSampling(ThreadState state, int target, float lr, bool addToOutput = true)
        {
            state.Gradient.Zero();

            float loss = 0f;
            for (int n = 0; n <= Data.NegativeSamplingCount; n++)
            {
                if (n == 0)
                {
                    loss += BinaryLogistic(state, target, true, lr, addToOutput);
                }
                else
                {
                    loss += BinaryLogistic(state, GetNegative(ref state.NegativePosition, target), false, lr, addToOutput);
                }
            }

            return loss;
        }

        private float ComputeOneVsAllLoss(ThreadState state, int[] targets, float lr, bool addToOutput = true)
        {
            state.Gradient.Zero();
            float loss = 0f;

            for (int i = 0; i < state.Output.Length; i++)
            {
                bool isMatch = targets.Contains(i);
                loss += BinaryLogistic(state, i, isMatch, lr, addToOutput);
            }

            return loss;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float BinaryLogistic(ThreadState state, int target, bool label, float lr, bool addToOutput = true)
        {
            var v = Wo.GetRow(target);
            Quantize(v);
            float score = state.Sigmoid(Wo.DotRow(state.Hidden, v));

            float alpha = lr * ((label ? 1.0f : 0f) - score);

            SIMD.MultiplyAndAdd(state.Gradient, Wo.GetRow(target), alpha);

            if (addToOutput)
            {
                Wo.AddToRow(state.Hidden, target, alpha);
            }

            if (label)
            {
                return -state.Log(score);
            }
            else
            {
                return -state.Log(1.0f - score);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int GetNegative(ref int NegativePosition, int target)
        {
            int negative;
            do
            {
                negative = NegativeTable[NegativePosition];
                NegativePosition = (NegativePosition + 1) % NegativeTable.Length;
            } while (target == negative);
            return negative;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeHierarchicalSoftMax(ThreadState state, int target, float lr, bool addToOutput = true)
        {
            float loss = 0.0f;
            state.Gradient.Zero();
            var binaryCode = HS_Codes[target];
            var pathToRoot = HS_Paths[target];
            for (int i = 0; i < pathToRoot.Count; i++)
            {
                loss += BinaryLogistic(state, pathToRoot[i], binaryCode[i], lr, addToOutput);
            }
            return loss;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeSoftmax(ThreadState state, int target, float lr, bool addToOutput = true)
        {
            ComputeOutputSoftmax(state);
            if (addToOutput)
            {
                state.Gradient.Zero();
                for (int i = 0; i < Wo.Rows; i++)
                {
                    float label = (i == target) ? 1.0f : 0.0f;
                    float alpha = lr * (label - state.Output[i]);
                    SIMD.Add(state.Gradient, Wo.GetRow(i));

                    Wo.AddToRow(state.Hidden, i, alpha);
                }
            }
            return (float)(-Math.Log(state.Output[target]));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ComputeOutputSoftmax(ThreadState state)
        {
            float z = 0.0f;
            ref float[] hidden = ref state.Hidden;
            ref float[] output = ref state.Output;

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = Wo.DotRow(hidden, i);
            }

            float max = SIMD.Max(output);

            for (int i = 0; i < output.Length; i++)
            {
                output[i] = (float)(Math.Exp(output[i] - max));
                z += output[i];
            }
            z = 1.0f / z;
            SIMD.Multiply(output, z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ComputeOutputBinaryLogistic(ThreadState state)
        {
            ref float[] hidden = ref state.Hidden;
            ref float[] output = ref state.Output;
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = Wo.DotRow(hidden, i);
                output[i] = state.Sigmoid(output[i]);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ComputeHidden(ThreadState state, Span<int> input)
        {
            float z = 1.0f / input.Length;
#if NETCOREAPP3_0
            Span<float> hidden = state.Hidden;
            hidden.Fill(0f);
#else
            float[] hidden = state.Hidden;
            hidden.Zero();
#endif
            foreach (var ix in input)
            {
                var v = Wi.GetRow(ix);
                Quantize(v);
                SIMD.Add(hidden, v);
            }
            SIMD.Multiply(hidden, z);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
#if NETCOREAPP3_0
        private void ComputeHiddenForPrediction(Span<float> hidden, Span<int> input, Span<float> extraVector)
        {
            hidden.Fill(0f);
#else
        private void ComputeHiddenForPrediction(float[] hidden, Span<int> input, Span<float> extraVector)
        {
            hidden.Zero();
#endif
            float z = 1.0f / (input.Length + 1);
            foreach (var ix in input)
            {
                var v = Wi.GetRow(ix);
                Quantize(v);
                SIMD.Add(hidden, v);
            }

            var ve = extraVector.ToArray();
            Quantize(ve);
            SIMD.Add(hidden, ve);

            SIMD.Multiply(hidden, z);
        }

        private ThreadState[] Initialize(InputData ID, CancellationToken token, VectorizerTrainingData previousTrainingCorpus = null)
        {
            bool retraining = (Data.EntryCount > 0);

            if (!retraining) //New model
            {
                Data.EntryHashToIndex = new Dictionary<uint, int>();
                Data.Entries = new Dictionary<int, Entry>();
                Data.SubwordHashToIndex = new Dictionary<uint, int>();
            }

            if (Data.LabelHashToIndex is null || Data.Labels is null)
            {
                Data.LabelHashToIndex = new Dictionary<uint, int>();
                Data.Labels = new Dictionary<int, Entry>();
            }

            //TODO: Finish implementing this
            //var multipleTokens = uniqueMultiple.Where(kv => kv.Value > Data.MinimumWordNgramsCounts).ToList();

            //Create labels
            if (Data.Type == ModelType.Supervised)
            {
                foreach (var lbl in ID.uniqueLabels)
                {
                    if (!Data.LabelHashToIndex.ContainsKey(lbl.Key))
                    {
                        Data.Labels.Add(Data.LabelCount, new Entry(lbl.Value.Value, (int)lbl.Value.Frequency, EntryType.Label, PartOfSpeech.X, Language.Any));
                        Data.LabelHashToIndex.Add(lbl.Key, Data.LabelCount);
                        Data.LabelCount++;
                    }
                    else
                    {
                        Logger.LogWarning("Hash colision between {ONE} and {TWO}", lbl.Value.Value, Data.Entries[Data.EntryHashToIndex[lbl.Key]].Word);
                    }
                }
            }

            token.ThrowIfCancellationRequested();

            if (Data.Type == ModelType.PVDBow || Data.Type == ModelType.PVDM)
            {
                foreach (var lbl in ID.docIDHashes)
                {
                    var hash = lbl.Value;
                    if (!Data.EntryHashToIndex.ContainsKey(hash))
                    {
                        var tk = ID.uniqueIDs[hash];
                        Data.Entries.Add(Data.EntryCount, new Entry(tk.Value, 1, EntryType.Label, PartOfSpeech.X, Language.Any));
                        Data.EntryHashToIndex.Add(hash, Data.EntryCount);
                        Data.EntryCount++;
                        Data.LabelCount++;
                    }
                    else
                    {
                        Logger.LogWarning("Hash colision between {ONE} and {TWO}", lbl.Value, Data.Entries[Data.EntryHashToIndex[hash]].Word);
                    }
                }
            }

            token.ThrowIfCancellationRequested();

            var orderedTokens = ID.uniqueTokens.ToList().Where(g => g.Value.Frequency >= Data.MinimumCount).OrderByDescending(g => g.Value.Frequency).ToList();

            //Create tokens
            foreach (var tk in orderedTokens)
            {
                if (!Data.EntryHashToIndex.ContainsKey(tk.Key))
                {
                    Data.Entries.Add(Data.EntryCount, new Entry((Data.IgnoreCase ? tk.Value.Value.ToLowerInvariant() : tk.Value.Value), (int)tk.Value.Frequency, EntryType.Word, tk.Value.POS, tk.Value.Language));
                    Data.EntryHashToIndex.Add(tk.Key, Data.EntryCount);
                    Data.EntryCount++;
                }
                else if (!retraining)
                {
                    Logger.LogWarning("Hash colision between {ONE} and {TWO}", tk.Value.Value, Data.Entries[Data.EntryHashToIndex[tk.Key]].Word);
                }
            }

            token.ThrowIfCancellationRequested();

            //Create corpus
            var trainingCorpus = new Line[ID.hashCorpus.Count];
            foreach (var di in ID.hashCorpus.Keys.ToArray())
            {
                var entries = ID.hashCorpus[di].Where(hash => Data.EntryHashToIndex.ContainsKey(hash))
                                            .Select(hash => Data.EntryHashToIndex[hash]).ToArray();

                var labels = new int[0];
                if (Data.Type == ModelType.Supervised)
                {
                    labels = ID.docLabelHashes.ContainsKey(di) ? ID.docLabelHashes[di].Select(hash => Data.LabelHashToIndex[hash]).ToArray() : new int[0];
                    GetWordNGrams(entries, create: true); //create entries on the index, but don't add them yet
                }
                else if (Data.Type == ModelType.PVDBow || Data.Type == ModelType.PVDM)
                {
                    if (Data.Type == ModelType.PVDM)
                    {
                        GetWordNGrams(entries, create: true); //create entries on the index, but don't add them to the InputData - they're re-computed on the fly during the PVDM
                    }

                    labels = ID.docIDHashes.ContainsKey(di) ? new int[1] { Data.EntryHashToIndex[ID.docIDHashes[di]] } : new int[0];
                }
                else if (Data.CBowUseWordNgrams && Data.Type == ModelType.CBow)
                {
                    GetWordNGrams(entries, create: true); //create entries on the index, but don't add them to the InputData - they're re-computed on the fly during training
                }

                trainingCorpus[di] = new Line(entries, labels);
                Interlocked.Add(ref NumberOfTokens, trainingCorpus[di].EntryIndexes.Length);
            }

            token.ThrowIfCancellationRequested();

            //Store lines for future model updates
            if (Data.StoreTrainingData)
            {
                if (previousTrainingCorpus is null || !retraining)
                {
                    previousTrainingCorpus = new VectorizerTrainingData();
                    previousTrainingCorpus.Lines.AddRange(trainingCorpus);
                }
                else
                {
                    //TODO: Do not load all data, just a few items that contain the same vocabulary as the new training corpus
                    var linesToReuse = previousTrainingCorpus.Lines.Shuffle().Take((int)(previousTrainingCorpus.Lines.Count * Data.ReusePreviousCorpusFactor));

                    NumberOfTokens += linesToReuse.Sum(l => l.EntryIndexes.Length);

                    previousTrainingCorpus.Lines.AddRange(trainingCorpus);

                    trainingCorpus = trainingCorpus.Concat(linesToReuse).ToArray();
                }

                if (!(previousTrainingCorpus is null))
                {
                    //Temporary fix for the fact that we can only serialize a single training corpus file of up to 2GB (max array size)
                    //We should eventually split the training corpus in multiple files!
                    const int maximumLines = 1_000_000;
                    if (previousTrainingCorpus.Lines.Count > maximumLines)
                    {
                        previousTrainingCorpus.Lines.RemoveRange(maximumLines, previousTrainingCorpus.Lines.Count - maximumLines);
                    }

                    using (var m = new Measure(Logger, "Storing training data for future training"))
                    {
                        DataStore.PutData(previousTrainingCorpus, Language, nameof(VectorizerTrainingData), Version, Tag + "-" + __DATA_FILE__, false);
                    }
                }
            }

            token.ThrowIfCancellationRequested();

            var corpusByThread = trainingCorpus.SplitIntoN(Data.Threads).ToArray();

            GradientLength = Data.Dimensions;
            HiddenLength = Data.Dimensions;

            if (Data.Type == ModelType.Supervised)
            {
                Wo = new Matrix(Data.LabelCount, Data.Dimensions);
                OutputLength = Data.LabelCount;
            }
            else
            {
                if (!retraining)
                {
                    Wo = new Matrix(Data.EntryCount, Data.Dimensions);
                }
                else
                {
                    Wo.ResizeAndFillRows(Data.EntryCount, 0f);
                }
                OutputLength = Data.EntryCount;
            }

            InitializeEntries();

            token.ThrowIfCancellationRequested();

            if (Data.Type == ModelType.Supervised)
            {
                //Need to add to the corpus the char and word ngrams
                for (int i = 0; i < trainingCorpus.Length; i++)
                {
                    var l = trainingCorpus[i];
                    var wordNgrams = GetWordNGrams(l.EntryIndexes, create: false);
                    var charNgrams = l.EntryIndexes.SelectMany(ei => GetEntrySubwords(ei).ToArray()).ToArray();

                    l.EntryIndexes = l.EntryIndexes.Concat(wordNgrams).Concat(charNgrams).ToArray();
                    trainingCorpus[i] = l;
                }
            }

            Logger.LogInformation("Found {WORDS} words, {LABELS} labels, {SUBWORDS} subwords and {TOKENS} tokens", Data.EntryCount - Data.LabelCount, Data.LabelCount, Data.SubwordCount, NumberOfTokens);

            using (var m = new Measure(Logger, "Creating input matrix"))
            {
                if (!retraining)
                {
                    Wi = new Matrix(Data.EntryCount + Data.SubwordCount + 1 /*(int)Data.Buckets*/, Data.Dimensions).Uniform(1.0f / Data.Dimensions);
                }
                else
                {
                    Wi.ResizeAndFillRows(Data.EntryCount + Data.SubwordCount + 1, 1.0f / Data.Dimensions);
                }
                m.SetOperations(Wi.Rows);
            }

            token.ThrowIfCancellationRequested();

            return corpusByThread.Select((c, i) => new ThreadState(c, HiddenLength, OutputLength, GradientLength, i, token)).ToArray();
        }

        private InputData ProcessDocuments(IEnumerable<IDocument> documents, Func<IToken, bool> ignorePattern, ParallelOptions parallelOptions)
        {
            var ID = new InputData();
            ID.docCount = -1;

            int ignoredDocuments = 0;

            parallelOptions = parallelOptions ?? new ParallelOptions();
            CancellationToken cancellationToken = parallelOptions?.CancellationToken ?? default;

            Parallel.ForEach(documents, parallelOptions, doc =>
            {
               if (cancellationToken.IsCancellationRequested) { return; }
               if (Language != Language.Any && doc.Language != Language) { Interlocked.Increment(ref ignoredDocuments); return; }

               if (Data.Type != ModelType.Supervised && doc.TokensCount < 2 * Data.ContextWindow) { return; } //Skip documents that are too small

                int docIndex = Interlocked.Increment(ref ID.docCount);

               var tokenHashes = new List<uint>(doc.TokensCount);

               foreach (var span in doc)
               {
                   var tokens = span.GetCapturedTokens().ToArray();

                   for (int i = 0; i < tokens.Length; i++)
                   {
                       if (ignorePattern is object && ignorePattern(tokens[i]))
                       {
                           continue;
                       }
                       uint hash = HashToken(tokens[i], Language);
                       ID.uniqueTokens.AddOrUpdate(hash, (key) => new SingleToken(tokens[i], Language) { Frequency = 1 }, (key, v) => { v.Frequency++; return v; });
                       tokenHashes.Add(hash);
                   }
               }

               tokenHashes.Add(_HashEOS_);
               ID.uniqueTokens.AddOrUpdate(_HashEOS_, (key) => new SingleToken(_EOS_, Language) { Frequency = 1 }, (key, v) => { v.Frequency++; return v; });

               ID.hashCorpus.TryAdd(docIndex, tokenHashes);

               if (Data.Type == ModelType.Supervised)
               {
                   if (doc.Labels.Any())
                   {
                       var lblHashes = new List<uint>(doc.Labels.Count);
                       foreach (var lbl in doc.Labels)
                       {
                           uint hash = HashLabel(lbl);
                           ID.uniqueLabels.AddOrUpdate(hash, (key) => new SingleToken(lbl, Language) { Frequency = 1 }, (key, v) => { v.Frequency++; return v; });
                           lblHashes.Add(hash);
                       }
                       ID.docLabelHashes.TryAdd(docIndex, lblHashes);
                   }
                   else
                   {
                       throw new Exception("Document has no labels attached");
                   }
               }
               if (Data.Type == ModelType.PVDBow || Data.Type == ModelType.PVDM)
               {
                   string id = doc.UID.ToString();
                    var hash = HashLabel(id);
                   ID.docIDHashes.TryAdd(docIndex, hash);
                   ID.uniqueIDs.TryAdd(hash, new SingleToken(id, Language));
               }

               if (docIndex % 10_000 == 0)
               {
                   Logger.LogInformation("Parsed {COUNT:n0} docs for training embeddings", docIndex);
               }
           });

            if (ignoredDocuments > 0)
            {
                Logger.LogWarning("Ignored {COUNT} documents that were in a different language from {LANGUAGE}", ignoredDocuments, Language);
            }

            cancellationToken.ThrowIfCancellationRequested();

            return ID;
        }

        private void InitializeEntries()
        {
            ////Create Context Window vectors for position-dependent weighting - see https://arxiv.org/pdf/1712.09405v1.pdf
            //if (!Data.ContextWindowIndexToIndex.Any())
            //{
            //    for (int c = -Data.ContextWindow; c <= Data.ContextWindow; c++)
            //    {
            //        var cwhash = Hash((_CONTEXTWINDOW_ + c.ToString()).AsSpan());
            //        var cwindex = TranslateNgramHashesToIndexes(new List<uint>() { cwhash }, Language.Any, create: true);
            //        Data.ContextWindowIndexToIndex.Add(c, (int)cwindex.Single());
            //    }
            //}

            using (var m = new Measure(Logger, "Initializing Entries"))
            {
                var EntrySubwords = new int[Data.EntryCount][];
                EntryDiscardProbability = new int[Data.EntryCount];

                foreach (var kv in Data.Entries)
                {
                    var subwords = new List<uint>();

                    if (Data.MaximumNgrams > 0 && kv.Value.Type == EntryType.Word)
                    {
                        subwords.AddRange(GetCharNgrams(kv.Value.Word));
                        if (kv.Value.POS != PartOfSpeech.NONE)
                        {
                            subwords.Add((uint)Data.EntryCount + (POS_Hashes[(int)kv.Value.POS] % Data.Buckets));
                        }
                        subwords = TranslateNgramHashesToIndexes(subwords, kv.Value.Language, create: true);
                    }

                    subwords.Add((uint)kv.Key);

                    EntrySubwords[kv.Key] = new int[subwords.Count];
                    for (int i = 0; i < subwords.Count; i++)
                    {
                        EntrySubwords[kv.Key][i] = (int)subwords[i];
                    }

                    double f = (double)kv.Value.Count / NumberOfTokens;
                    EntryDiscardProbability[kv.Key] = (int)(System.Int32.MaxValue * (1.0 - Math.Sqrt(Data.SamplingThreshold / f)));// + SamplingThreshold / f;
                }

                if (Data.Type == ModelType.Supervised)
                {
                    SetTargetCounts(Data.Labels.Values.Select(e => e.Count).ToList());
                }
                else if (Data.Type == ModelType.PVDM || Data.Type == ModelType.PVDBow)
                {
                    SetTargetCounts(Data.Entries.Values.Select(e => e.Count).ToList());
                }
                else
                {
                    SetTargetCounts(Data.Entries.Values.Where(e => e.Type == EntryType.Word).Select(e => e.Count).ToList());
                }

                int usedBuckets = EntrySubwords.SelectMany(e => e).Distinct().Count();

                var totalSize = EntrySubwords.Sum(e => e.Length);

                int p = 0;
                EntrySubwordsFlatten = new int[totalSize];
                EntrySubwordsBegin = new int[EntrySubwords.Length];
                EntrySubwordsLength = new int[EntrySubwords.Length];

                for (int i = 0; i < EntrySubwords.Length; i++)
                {
                    int l = EntrySubwords[i].Length;
                    EntrySubwords[i].AsSpan().CopyTo(EntrySubwordsFlatten.AsSpan().Slice(p, l));
                    EntrySubwordsBegin[i] = p;
                    EntrySubwordsLength[i] = l;
                    p += l;
                }

                m.SetOperations(usedBuckets);
            }

             ThreadStatePool = new ObjectPool<ThreadState>(() => new ThreadState(new Line[0], HiddenLength, OutputLength, GradientLength, -1, CancellationToken.None), 2);
        }

        private List<uint> TranslateNgramHashesToIndexes(List<uint> hashes, Language language, bool create = true)
        {
            var indexes = new List<uint>(hashes.Count);
            //var offset = (uint)Data.LanguageOffset[language];
            for (int i = 0; i < hashes.Count; i++)
            {
                if (Data.SubwordHashToIndex.TryGetValue(hashes[i]/* + offset*/, out int index))
                {
                    indexes.Add((uint)index);
                }
                else if (create)
                {
                    index = Interlocked.Increment(ref Data.SubwordCount) + Data.EntryCount;
                    Data.SubwordHashToIndex.Add(hashes[i]/* + offset*/, index);
                    indexes.Add((uint)index);
                }
            }
            return indexes;
        }

        public int[] GetWordNGrams(Span<int> list, bool create)
        {
            int len = list.Length;
            var hashes = new List<uint>();
            for (int w = 0; w < len; w++)
            {
                for (int contextSize = 2; contextSize <= Data.MaximumWordNgrams; contextSize++)
                {
                    int hash = list[w];
                    bool add = false;
                    for (int c = 1; c < contextSize; c++)
                    {
                        if (w + c >= 0 && w + c < len)
                        {
                            hash = Hashes.CombineWeak(hash, list[w + c]);
                            add = true;
                        }
                    }
                    if (add) hashes.Add((uint)Data.EntryCount + (uint)hash % Data.Buckets);
                }
            }
            hashes = TranslateNgramHashesToIndexes(hashes, Language.Any, create);
            return hashes.Select(h => (int)h).ToArray();
        }

        private void AppendWordNGrams(List<int> list, bool create)
        {
            if (Data.MaximumWordNgrams < 2) { return; }
            int len = list.Count;
            var hashes = new List<uint>();
            for (int w = 0; w < len; w++)
            {
                for (int contextSize = 2; contextSize <= Data.MaximumWordNgrams; contextSize++)
                {
                    int hash = list[w];
                    bool add = false;
                    for (int c = 1; c < contextSize; c++)
                    {
                        if (w + c >= 0 && w + c < len)
                        {
                            hash = Hashes.CombineWeak(hash, list[w + c]);
                            add = true;
                        }
                    }
                    if (add) hashes.Add((uint)Data.EntryCount + (uint)hash % Data.Buckets);
                }
            }
            hashes = TranslateNgramHashesToIndexes(hashes, Language.Any, create);
            list.AddRange(hashes.Select(h => (int)h));
        }

        public uint HashToken(IToken tk, Language language)
        {
            return HashToken(tk.ValueAsSpan, tk.POS, language);
        }

        public uint HashToken(ReadOnlySpan<char> value, PartOfSpeech pos, Language language)
        {
            if (Data.IgnoreCase)
            {
                return (uint)Hashes.CombineWeak(Hashes.CombineWeak(Hashes.IgnoreCaseHash32(value), POS_Hashes[(int)pos]), Language_Hashes[(int)language]);
            }
            else
            {
                return (uint)Hashes.CombineWeak(Hashes.CombineWeak(Hashes.CaseSensitiveHash32(value), POS_Hashes[(int)pos]), Language_Hashes[(int)language]);
            }
        }

        private static uint HashLabel(string label)
        {
            //Label hash is always case-dependent
            return (uint)Hashes.CombineWeak(Hashes.CaseSensitiveHash32(label), _LABEL_HASH_);
        }

        internal List<uint> GetCharNgrams(string word)
        {
            bool addBOW = true;
            bool addEOW = true;
            bool ignoreCase = false;
            int minNgrams = Data.MinimumNgrams;
            int maxNgrams = Data.MaximumNgrams;
            uint buckets = Data.Buckets;

            var ngrams = new List<uint>();
            var ngram = new char[maxNgrams];
            int cur = 0;
            var chars = word.AsSpan();
            int length = chars.Length;
            int bow = -1, eow = -1, d = 0;

            if (addBOW)
            {
                bow = 0; length++; d++;
            }

            if (addEOW)
            {
                eow = length;
                length++;
            }

            for (int i = 0; i < length; i++)
            {
                cur = 0;

                for (int j = i, n = 1; j < length && n <= maxNgrams; n++)
                {
                    if (j == bow)
                    {
                        ngram[cur] = _BOW_;
                    }
                    else if (j == eow)
                    {
                        ngram[cur] = _EOW_;
                    }
                    else
                    {
                        ngram[cur] = (ignoreCase ? char.ToLowerInvariant(chars[j - d]) : chars[j - d]);
                    }
                    cur++; j++;

                    if (n >= minNgrams && !(n == 1 && (i == 0 || j == word.Length)))
                    {
                        var hash = (Hash(ngram.AsSpan().Slice(0, cur)) % buckets);
                        ngrams.Add((uint)Data.EntryCount + hash);
                    }
                }
            }
            return ngrams;
        }

        public static uint Hash(ReadOnlySpan<char> word)
        {
            return (uint)word.CaseSensitiveHash32();
        }

        public class VectorizerTrainingData : StorableObjectData
        {
            public List<Line> Lines = new List<Line>();
        }

        public enum ModelType
        {
            CBow,
            Skipgram,
            Supervised,
            PVDM,
            PVDBow
        }

        public class HSNode
        {
            public int parent;
            public int left;
            public int right;
            public long count;
            public bool binary;
        };

        public class InputData
        {
            public ConcurrentDictionary<int, uint> docIDHashes = new ConcurrentDictionary<int, uint>();
            public ConcurrentDictionary<int, List<uint>> hashCorpus = new ConcurrentDictionary<int, List<uint>>();
            public ConcurrentDictionary<int, List<uint>> docLabelHashes = new ConcurrentDictionary<int, List<uint>>();
            public ConcurrentDictionary<uint, SingleToken> uniqueLabels = new ConcurrentDictionary<uint, SingleToken>();
            public ConcurrentDictionary<uint, SingleToken> uniqueIDs = new ConcurrentDictionary<uint, SingleToken>();
            public ConcurrentDictionary<uint, SingleToken> uniqueTokens = new ConcurrentDictionary<uint, SingleToken>();
            public int docCount;
        }
    }
}