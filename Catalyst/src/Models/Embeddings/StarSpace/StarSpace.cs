using UID;
using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Catalyst.Models
{
    public class StarSpace : StorableObject<StarSpace, StarSpaceModel>, ITrainableModel
    {
        public const char _BOW_ = '<';
        public const char _EOW_ = '>';
        public const string _EOS_ = "</s>";
        public static uint _HashEOS_; //Initialized on creation, as it depends on POS being already initialized - otherwise might run into a race condition on who's created first
        private static readonly uint[] POS_Hashes = Enum.GetValues(typeof(PartOfSpeech)).Cast<PartOfSpeech>().Select(pos => (uint)Hashes.CaseSensitiveHash32(pos.ToString())).ToArray();
        private static readonly uint[] Language_Hashes = Enum.GetValues(typeof(Language)).Cast<Language>().Select(lang => (uint)Hashes.CaseSensitiveHash32(lang.ToString())).ToArray();

        public TrainingHistory TrainingHistory => Data.TrainingHistory;

        private SharedState Shared;

        public event EventHandler<TrainingUpdate> TrainingStatus;

        private StarSpace(Language language, int version, string tag) : base(language, version, tag, compress: true)
        {
            _HashEOS_ = HashToken(_EOS_.AsSpan(), PartOfSpeech.NONE, Language.Any);
        }

        public StarSpace(Language language, int version, string tag, ModelType modelType) : this(language, version, tag)
        {
            Data.Type = modelType;
            Data.ThreadCount = Environment.ProcessorCount;
            Data.ThreadPriority = ThreadPriority.Normal;
        }

        public new static async Task<StarSpace> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new StarSpace(language, version, tag);
            await a.LoadDataAsync();

            (Stream lhsStream, Stream rhsStream) = await a.GetMatrixStreamsAsync();

            a.Shared = new SharedState();

            lhsStream = TryPreloadInMemory(lhsStream, a.GetStoredObjectInfo().ToString() + "-lhs");
            a.Shared.LHSEmbeddings = Matrix.FromStream(lhsStream, a.Data.VectorQuantization);

            rhsStream = TryPreloadInMemory(rhsStream, a.GetStoredObjectInfo().ToString() + "-rhs");
            a.Shared.RHSEmbeddings = Matrix.FromStream(rhsStream, a.Data.VectorQuantization);

            lhsStream.Close();
            rhsStream.Close();

            return a;
        }

        private async Task<(LockedStream lhs, LockedStream rhs)> GetMatrixStreamsAsync()
        {
            var lhs = await DataStore.OpenReadAsync(Language, nameof(StarSpaceModel) + "-Matrix", Version, Tag + "-lhs");
            var rhs = await DataStore.OpenReadAsync(Language, nameof(StarSpaceModel) + "-Matrix", Version, Tag + "-rhs");
            return (lhs, rhs);
        }

        private static Stream TryPreloadInMemory(Stream s, string name)
        {
            if (s is object && s.Length > 0 && s.Length < (int.MaxValue / 2))
            {
                try
                {
                    var s2 = new MemoryStream((int)s.Length);
                    s.CopyTo(s2);
                    s.Dispose();
                    s2.Seek(0, SeekOrigin.Begin);
                    s = s2;
                }
                catch
                {
                    //Ignore
                }
            }

            return s;
        }

        public override async Task StoreAsync()
        {
            var lhsStream = await DataStore.OpenWriteAsync(Language, nameof(StarSpaceModel) + "-Matrix", Version, Tag + "-lhs");
            var rhsStream = await DataStore.OpenWriteAsync(Language, nameof(StarSpaceModel) + "-Matrix", Version, Tag + "-rhs");

            Shared.LHSEmbeddings.ToStream(lhsStream, Data.VectorQuantization);
            Shared.RHSEmbeddings.ToStream(rhsStream, Data.VectorQuantization);

            lhsStream.Close();
            rhsStream.Close();

            await base.StoreAsync();
        }

        public new static async Task<bool> DeleteAsync(Language language, int version, string tag)
        {
            var a = new FastText(language, version, tag);
            bool deleted = false;
            deleted |= await DataStore.DeleteAsync(language, nameof(StarSpaceModel) + "-Matrix", version, tag + "-lhs");
            deleted |= await DataStore.DeleteAsync(language, nameof(StarSpaceModel) + "-Matrix", version, tag + "-rhs");
            deleted |= await a.DeleteDataAsync();
            return deleted;
        }

        public void Train(IEnumerable<IDocument> documents, Func<IToken, bool> ignorePattern = null, ParallelOptions parallelOptions = default)
        {
            InputData inputData;

            Data.InputType = InputType.LabeledDocuments;

            CancellationToken cancellationToken = parallelOptions?.CancellationToken ?? default;

            using (var scope = Logger.BeginScope($"Training StarSpace '{Tag}' of type {Data.Type} from documents"))
            {
                using (var m = new Measure(Logger, "Document parsing", 1))
                {
                    inputData = ProcessDocuments(documents, ignorePattern, parallelOptions);
                    m.SetOperations(inputData.docCount);
                }

                using (var m = new Measure(Logger, "Training vector model " + (Vector.IsHardwareAccelerated ? "using hardware acceleration [" + Vector<float>.Count + "]" : "without hardware acceleration"), inputData.docCount))
                {
                    DoTraining(inputData, cancellationToken);
                }
            }
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

                if (Data.Type == ModelType.WordEmbeddings && doc.TokensCount < 2 * Data.ContextWindow) { return; } //Skip documents that are too small

                int docIndex = Interlocked.Increment(ref ID.docCount);

                var docParse = new List<List<Base>>();

                foreach (var span in doc)
                {
                    var tokens = span.GetTokenized().ToArray();
                    var spanParse = new List<Base>();
                    for (int i = 0; i < tokens.Length; i++)
                    {
                        if (ignorePattern is object && ignorePattern(tokens[i]))
                        {
                            continue;
                        }
                        uint hash = HashToken(tokens[i], Language);
                        ID.uniqueTokens.AddOrUpdate(hash, (key) => new SingleToken(tokens[i], Language) { Frequency = 1 }, (key, v) => { v.Frequency++; return v; });
                        spanParse.Add(new Base(-1, hash, 1f));
                    }

                    if (Data.WordNGrams > 1)
                    {
                        throw new NotImplementedException(); //TODO: Add word ngrams, see parser.cpp -> addNgrams
                    }

                    spanParse.Add(new Base(-1, _HashEOS_, 1f));
                    docParse.Add(spanParse);
                }

                ID.uniqueTokens.AddOrUpdate(_HashEOS_, (key) => new SingleToken(_EOS_, Language) { Frequency = 1 }, (key, v) => { v.Frequency++; return v; });

                ID.Corpus.TryAdd(docIndex, docParse);

                //TODO: if using labels, need to also add the label of the document

                if (docIndex % 10_000 == 0)
                {
                    Logger.LogInformation("Parsed {COUNT:n0} docs for training StarSpace embeddings", docIndex);
                }
            });

            if (ignoredDocuments > 0)
            {
                Logger.LogWarning("Ignored {COUNT} documents that were in a different language from {LANGUAGE}", ignoredDocuments, Language);
            }

            cancellationToken.ThrowIfCancellationRequested();

            return ID;
        }

        public void DoTraining(InputData inputData, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ThreadState[] threadState;
            SharedState sharedState;

            using (var m = new Measure(Logger, "Initializing private thread states for training", 1))
            {
                int nLabels = 0; //TODO: Missing labels!
                (sharedState, threadState) = InitializeStates(inputData, cancellationToken);
                sharedState.InitializeModelWeights(Data.Dimensions, inputData.uniqueTokens.Count, nLabels, (Data.WordNGrams > 1 ? Data.Buckets : 0), Data.ShareEmbeddings, Data.AdaGrad, Data.InitializationStandardDeviation);
            }

            float rate = Data.LearningRate;
            float decrPerEpoch = (rate - 1e-9f) / Data.Epoch;
            int impatience = 0;
            float best_valid_err = 1e9f;

            var trainingHistory = new TrainingHistory();

            using (var m = new Measure(Logger, "Training", Data.Epoch))
            {
                for (int epoch = 0; epoch < Data.Epoch; epoch++)
                {
                    var threads = threadState.Select(mps =>
                    {
                        mps.Rate = rate;
                        mps.FinishRate = rate - decrPerEpoch;
                        mps.Epoch = epoch;
                        mps.TrainingHistory = epoch == 0 ? trainingHistory : null;
                        var t = new Thread(() => ThreadTrain(mps));
                        t.Priority = Data.ThreadPriority;
                        t.Start();
                        return t;
                    }).ToArray();
                    foreach (var t in threads) { t.Join(); }

                    rate -= decrPerEpoch;
                    cancellationToken.ThrowIfCancellationRequested(); //If the training was canceled, all threads will return, so we throw here
                }
                trainingHistory.ElapsedTime = TimeSpan.FromSeconds(m.ElapsedSeconds);
            }
            Data.TrainingHistory = trainingHistory;
            Data.IsTrained = true;
            Shared = sharedState;
        }

        public float[] PredictDocumentVector(IDocument doc)
        {
            var docParse = new List<Base>();

            foreach (var span in doc)
            {
                foreach (var v in span.GetTokenized())
                {
                    //if (ignorePattern is object && ignorePattern(tokens[i]))
                    //{
                    //    continue;
                    //}
                    uint hash = HashToken(v, Language);
                    if (Data.EntryHashToIndex.TryGetValue(hash, out var ix))
                    {
                        docParse.Add(new Base(ix, hash, 1f));
                    }
                }

                if (Data.WordNGrams > 1)
                {
                    throw new NotImplementedException(); //TODO: Add word ngrams, see parser.cpp -> addNgrams
                }
            }

            docParse.Add(new Base(-1, _HashEOS_, 1f));

            return PredictVectorFromParse(docParse);
        }

        public float[] PredictVectorFromParse(List<Base> parse)
        {
            //TODO: check if this is correct - not sure what this vector is supposed to be here, as we are now adding all the features to the same basket
            float[] vector = new float[Data.Dimensions];

            ProjectLHS(Shared, parse, vector);

            return vector;
        }

        private (SharedState sharedState, ThreadState[] threadState) InitializeStates(InputData inputData, CancellationToken cancellationToken)
        {
            Data.EntryHashToIndex = new Dictionary<uint, int>();
            Data.Entries = new Dictionary<int, FastText.Entry>();

            var sharedState = new SharedState();

            var orderedTokens = inputData.uniqueTokens.ToList().Where(g => g.Value.Frequency >= Data.MinimumCount).OrderByDescending(g => g.Value.Frequency).ToList();

            //Create tokens
            foreach (var tk in orderedTokens)
            {
                if (!Data.EntryHashToIndex.ContainsKey(tk.Key))
                {
                    Data.Entries.Add(Data.EntryCount, new FastText.Entry((Data.InvariantCase ? tk.Value.Value.ToLowerInvariant() : tk.Value.Value), (int)tk.Value.Frequency, FastText.EntryType.Word, tk.Value.POS, tk.Value.Language));
                    Data.EntryHashToIndex.Add(tk.Key, Data.EntryCount);
                    Data.EntryCount++;
                }
                else
                {
                    Logger.LogWarning("Hash colision between {ONE} and {TWO}", tk.Value.Value, Data.Entries[Data.EntryHashToIndex[tk.Key]].Word);
                }
            }

            List<ParseResults>[] trainingCorpus = Enumerable.Range(0, Data.ThreadCount).Select(i => new List<ParseResults>()).ToArray();

            int k = 0;
            foreach (var di in inputData.Corpus.Keys)
            {
                //TODO: different parsing mode depending on Data.InputType
                if (Data.InputType == InputType.FastText) { throw new NotImplementedException(); }

                var cur = inputData.Corpus[di];
                for (int i = 0; i < cur.Count; i++)
                {
                    List<Base> c = cur[i];
                    for (int j = 0; j < c.Count; j++)
                    {
                        Base b = c[j];
                        if (Data.EntryHashToIndex.TryGetValue(b.Hash, out var id))
                        {
                            c[j] = new Base(id, b.Hash, b.Weight);
                        }
                    }
                }
                cur.ForEach(l => l.RemoveAll(b => b.ID < 0));
                ParseResults parseResult = new ParseResults()
                {
                    RHSFeatures = cur.Select(c => c.ToArray()).ToArray()
                };

                if (Check(parseResult))
                {
                    trainingCorpus[k % trainingCorpus.Length].Add(parseResult);
                }

                k++;
            }

            Logger.LogInformation("Found {COUNT} examples for training", trainingCorpus.Sum(tc => tc.Count));

            return (sharedState, trainingCorpus.Select((tc, i) => new ThreadState(tc.ToArray(), i, cancellationToken, sharedState, i == 0 ? new Measure(Logger, "Training StarSpace Model") : null, Data.BatchSize, Data.NegativeSamplingSearchLimit, Data.Dimensions)).ToArray());
        }

        private bool Check(ParseResults example)
        {
            if (Data.InputType == InputType.FastText)
            {
                if (Data.Type == ModelType.TagSpace) //or any other == 0
                {
                    // require lhs and rhs
                    return example.RHSTokens.Count > 0 && example.LHSTokens.Count > 0;
                }
                if (Data.Type == ModelType.WordEmbeddings)
                {
                    // only requires lhs.
                    return example.LHSTokens.Count > 0;
                }
                else
                {
                    // lhs is not required, but rhs should contain at least 2 example
                    return example.RHSTokens.Count > 1;
                }
            }
            else
            {
                if (Data.Type == 0)
                {
                    return (example.LHSTokens.Count > 0) && (example.RHSFeatures.Length > 0);
                }
                else
                {
                    return example.RHSFeatures.Length > 1; // need to have at least two examples
                }
            }
        }

        private void ThreadTrain(ThreadState state)
        {
            // If we decrement after *every* sample, precision causes us to lose the update.
            int kDecrStep = 1000;
            var numSamples = state.Corpus.Length;
            var repInterval = (int)(numSamples / 20); //Reports every 5%
            var sw = Stopwatch.StartNew();
            float decrPerKSample = (state.Rate - state.FinishRate) / (numSamples / kDecrStep);
            int negSearchLimit = Math.Min(numSamples, Data.NegativeSamplingSearchLimit);

            //todo: access state.corpus in random order - i.e. create array of indices, shuffle and use this to index "i"

            // Compute word negatives
            if (Data.Type == ModelType.WordEmbeddings || Data.TrainWordEmbeddings)
            {
                state.InitializeWordNegatives();
            }

            var examples = new List<ParseResults>();
            var wordExamples = new List<ParseResults>();

            var indexes = Enumerable.Range(0, numSamples).ToList();
            indexes.Shuffle();

            for (int ix = 0; ix < numSamples; ix++)
            {
                int i = indexes[ix];
                float thisLoss = 0f;
                if (Data.Type == ModelType.WordEmbeddings || Data.TrainWordEmbeddings)
                {
                    GetWordExample(state, i, examples);
                    wordExamples.Clear();
                    for (int j = 0; j < examples.Count; j++)
                    {
                        wordExamples.Add(examples[i]);
                        if (wordExamples.Count >= Data.BatchSize || j == examples.Count - 1)
                        {
                            if (Data.LossType == LossType.SoftMax)
                            {
                                thisLoss = TrainNLLBatch(state, wordExamples, negSearchLimit, state.Rate, true);
                            }
                            else
                            {
                                thisLoss = TrainOneBatch(state, wordExamples, negSearchLimit, state.Rate, true);
                            }
                            wordExamples.Clear();
                            state.Counts++;
                            state.Loss += thisLoss;
                        }
                    }
                }
                if (Data.Type != ModelType.WordEmbeddings)
                {
                    ParseResults ex = GetExample(state, i);
                    if (ex.LHSTokens.Count == 0 || ex.RHSTokens.Count == 0)
                    {
                        continue;
                    }
                    examples.Add(ex);
                    if (examples.Count >= Data.BatchSize || i == numSamples - 1)
                    {
                        if (Data.LossType == LossType.SoftMax)
                        {
                            thisLoss = TrainNLLBatch(state, examples, negSearchLimit, state.Rate, false);
                        }
                        else
                        {
                            thisLoss = TrainOneBatch(state, examples, negSearchLimit, state.Rate, false);
                        }
                        examples.Clear();
                        state.Counts++;
                        state.Loss += thisLoss;
                    }
                }

                // update rate racily.
                if ((ix % kDecrStep) == (kDecrStep - 1))
                {
                    state.Rate -= decrPerKSample;
                }

                if (state.ThreadID == 0 && i % repInterval == 0 && state.Counts > 0)
                {
                    var curLos = state.Loss / state.Counts;
                    state.Measure.EmitPartial($"E:{state.Epoch} P:{100f * ((float)i / numSamples):n2}% L:{curLos:n8}");

                    var update = new TrainingUpdate().At(state.Epoch + ((float)i / numSamples), Data.Epoch, curLos).Processed(i, sw.Elapsed);
                    TrainingStatus?.Invoke(this, update);
                    state.TrainingHistory.Append(update);
                }
            }

            //long localTokenCount = 0;
            //Stopwatch Watch = null;
            //if (mps.ThreadID == 0) { Watch = Stopwatch.StartNew(); }
            //float progress = 0f, lr = Data.LearningRate;

            //float baseLR = Data.LearningRate / 200;

            //float nextProgressReport = 0f;
            //for (int epoch = 0; epoch < Data.Epoch; epoch++)
            //{
            //    if (mps.CancellationToken.IsCancellationRequested) { return; } //Cancelled the training, so return from the thread

            //    for (int i = 0; i < mps.Corpus.Length; i++)
            //    {
            //        localTokenCount += mps.Corpus[i].EntryIndexes.Length;

            //        switch (Data.Type)
            //        {
            //            case VectorModelType.CBow: { CBow(ref mps, ref mps.Corpus[i].EntryIndexes, lr); break; }
            //            case VectorModelType.Skipgram: { Skipgram(ref mps, ref mps.Corpus[i].EntryIndexes, lr); break; }
            //            case VectorModelType.Supervised: { Supervised(ref mps, ref mps.Corpus[i], lr); break; }
            //            case VectorModelType.PVDM: { PVDM(ref mps, ref mps.Corpus[i], lr); break; }
            //            case VectorModelType.PVDBow: { PVDBow(ref mps, ref mps.Corpus[i], lr); break; }
            //        }

            //        if (localTokenCount > Data.LearningRateUpdateRate)
            //        {
            //            progress = (float)(TokenCount) / (Data.Epoch * NumberOfTokens);

            //            var x10 = (float)(TokenCount) / (10 * NumberOfTokens);

            //            //lr = Data.LearningRate * (1.0f - progress);
            //            //plot abs(cos(x))*0.98^x from x = [0,100]
            //            //lr = (float)(baseLR + (Data.LearningRate - baseLR) * (0.5 + 0.5 * Math.Sin(100 * progress))); //Cyclic loss rate
            //            lr = (float)(baseLR + (Data.LearningRate - baseLR) * Math.Abs(Math.Cos(200 * x10)) * Math.Pow(0.98, 100 * x10)); //Cyclic loss rate, scaled for 10 epoch

            //            Interlocked.Add(ref TokenCount, localTokenCount);
            //            Interlocked.Add(ref PartialTokenCount, localTokenCount);

            //            localTokenCount = 0;

            //            if (mps.ThreadID == 0 && progress > nextProgressReport)
            //            {
            //                nextProgressReport += 0.01f; //Report every 1%
            //                var loss = mps.GetLoss();
            //                var ws = (double)(Interlocked.Exchange(ref PartialTokenCount, 0)) / Watch.Elapsed.TotalSeconds;
            //                Watch.Restart();
            //                var wst = ws / Data.Threads;

            //                Logger.LogInformation("At {PROGRESS}%, w/s/t: {WST}, w/s: {WS}, loss at epoch {EPOCH}/{MAXEPOCH}: {LOSS}", (int)(progress * 100f), (int)wst, (int)ws, epoch + 1, Data.Epoch, loss);
            //            }
            //        }
            //    }
            //}
        }

        private float TrainOneBatch(ThreadState state, List<ParseResults> batch_exs, int negSearchLimit, float rate0, bool trainWord)
        {
            var batch_sz = batch_exs.Count;
            var cols = Data.Dimensions;
            var total_loss = 0f;

            var negMean = new float[batch_sz][];
            var update_flag = new bool[batch_sz][];

            Span<float> posSim = stackalloc float[batch_sz];
            Span<float> labelRate = stackalloc float[batch_sz];
            Span<float> loss = stackalloc float[batch_sz];
            Span<int> num_negs = stackalloc int[batch_sz];

            for (int i = 0; i < batch_sz; i++) { labelRate[i] = rate0; }

            for (int i = 0; i < batch_sz; i++)
            {
                ProjectLHS(state.Shared, batch_exs[i].LHSTokens, state.lhs[i]);
                ProjectRHS(state.Shared, batch_exs[i].RHSTokens, state.rhsP[i]);
                posSim[i] = Similarity(ref state.lhs[i], ref state.rhsP[i]);
            }

            state.batch_negLabels.Clear();

            for (int i = 0; i < negSearchLimit; i++)
            {
                var negLabels = new List<Base>();
                if (trainWord)
                {
                    GetRandomWord(state, negLabels);
                }
                else
                {
                    GetRandomRHS(state, negLabels);
                }
                ProjectRHS(state.Shared, negLabels, state.rhsN[i]);
                state.batch_negLabels.Add(negLabels);
            }

            // Select negative examples
            for (int i = 0; i < batch_sz; i++)
            {
                num_negs[i] = 0;
                loss[i] = 0f;
                negMean[i] = new float[cols];
                update_flag[i] = new bool[negSearchLimit];

                for (int j = 0; j < negSearchLimit; j++)
                {
                    state.nRate[i][j] = 0.0f;
                    if (batch_exs[i].RHSTokens == state.batch_negLabels[j])
                    {
                        continue;
                    }
                    var thisLoss = TripleLoss(posSim[i], Similarity(ref state.lhs[i], ref state.rhsN[j]));
                    if (thisLoss > 0.0)
                    {
                        num_negs[i]++;
                        loss[i] += thisLoss;
                        SIMD.Add(negMean[i], state.rhsN[j]);
                        update_flag[i][j] = true;
                        if (num_negs[i] == Data.MaximumNegativeSamples) { break; }
                    }
                }
                if (num_negs[i] == 0) { continue; }

                loss[i] /= negSearchLimit;
                SIMD.Multiply(negMean[i], 1f / num_negs[i]);
                total_loss += loss[i];

                // gradW for i
                SIMD.MultiplyAndAdd(negMean[i], state.rhsP[i], -1f);
                for (int j = 0; j < negSearchLimit; j++)
                {
                    if (update_flag[i][j])
                    {
                        state.nRate[i][j] = rate0 / num_negs[i];
                    }
                }
            }

            // Couldn't find a negative example given reasonable effort, so
            // give up.
            if (total_loss == 0f) return 0f;
            if (rate0 == 0f) return total_loss;

            // Let w be the average of the input features, t+ be the positive example and t- be the average of the negative examples. Our error E is:
            //
            //    E = k - dot(w, t+) + dot(w, t-)
            //
            // Differentiating term-by-term we get:
            //
            //     dE / dw  = t- - t+
            //     dE / dt- = w
            //     dE / dt+ = -w
            //
            // gradW = \sum_i t_i- - t+. We're done with negMean, so reuse it.

            Backward(state, batch_exs, state.batch_negLabels, ref negMean, ref state.lhs, num_negs, rate0, labelRate, ref state.nRate);

            return total_loss;
        }

        private void Backward(ThreadState state, List<ParseResults> batch_exs, List<List<Base>> batch_negLabels, ref float[][] gradW, ref float[][] lhs, Span<int> num_negs, float rate_lhs, Span<float> rate_rhsP, ref float[][] nRate)
        {
            var cols = Data.Dimensions;

#if NETCOREAPP3_0
            void Update(Span<float> dest, ReadOnlySpan<float> src, float rate, float weight, Span<float> adagradWeight, int idx)
            {
                if (Data.AdaGrad)
                {
                    adagradWeight[idx] += weight / cols;
                    rate /= (float)Math.Sqrt(adagradWeight[idx] + 1e-6f);
                }
                SIMD.MultiplyAndAdd(dest, src, -rate);
            }
#else
            void Update(float[] dest, float[] src, float rate, float weight, Span<float> adagradWeight, int idx)
            {
                if (Data.AdaGrad)
                {
                    adagradWeight[idx] += weight / cols;
                    rate /= (float)Math.Sqrt(adagradWeight[idx] + 1e-6f);
                }
                SIMD.MultiplyAndAdd(dest, src, -rate);
            }
#endif
            var batch_sz = batch_exs.Count;
            var n1 = new float[batch_sz];
            var n2 = new float[batch_sz];
            if (Data.AdaGrad)
            {
                for (int i = 0; i < batch_sz; i++)
                {
                    if (num_negs[i] > 0)
                    {
                        n1[i] = SIMD.DotProduct(gradW[i], gradW[i]);
                        n2[i] = SIMD.DotProduct(lhs[i], lhs[i]);
                    }
                }
            }
            // Update input items.
            // Update positive example.
            for (int i = 0; i < batch_sz; i++)
            {
                if (num_negs[i] > 0)
                {
                    var items = batch_exs[i].LHSTokens;
                    var labels = batch_exs[i].RHSTokens;
                    foreach (var w in items)
                    {
                        var row = state.Shared.LHSEmbeddings.GetRow(w.ID);
                        Update(row, gradW[i], rate_lhs * w.Weight, n1[i], state.Shared.LHSUpdates, w.ID);
                    }
                    foreach (var la in labels)
                    {
                        var row = state.Shared.RHSEmbeddings.GetRow(la.ID);
                        Update(row, lhs[i], rate_rhsP[i] * la.Weight, n2[i], state.Shared.RHSUpdates, la.ID);
                    }
                }
            }

            // Update negative example
            for (int j = 0; j < batch_negLabels.Count; j++)
            {
                for (int i = 0; i < batch_sz; i++)
                {
                    if (Math.Abs(nRate[i][j]) > 1e-8)
                    {
                        foreach (var la in batch_negLabels[j])
                        {
                            var row = state.Shared.RHSEmbeddings.GetRow(la.ID);
                            Update(row, lhs[i], nRate[i][j] * la.Weight, n2[i], state.Shared.RHSUpdates, la.ID);
                        }
                    }
                }
            }
        }

        private void GetRandomRHS(ThreadState state, List<Base> results)
        {
            results.Clear();
            var ex = state.Corpus[ThreadSafeFastRandom.Next(state.Corpus.Length)];
            if (Data.InputType == InputType.FastText)
            {
                // Randomly sample one example and randomly sample a label from this example. The result is usually used as negative samples in training
                int r = ThreadSafeFastRandom.Next(ex.RHSTokens.Count);
                if (Data.Type == ModelType.ArticleSpace)
                {
                    for (int i = 0; i < ex.RHSTokens.Count; i++)
                    {
                        if (i != r)
                        {
                            results.Add(ex.RHSTokens[i]);
                        }
                    }
                }
                else
                {
                    results.Add(ex.RHSTokens[r]);
                }
            }
            else
            {
                int r = ThreadSafeFastRandom.Next(ex.RHSFeatures.Length);
                if (Data.Type == ModelType.ArticleSpace) // 2
                {
                    // pick one random, the rest is rhs features
                    for (int i = 0; i < ex.RHSFeatures.Length; i++)
                    {
                        if (i != r)
                        {
                            results.AddRange(ex.RHSFeatures[i]); //TODO: implement dropout , args_->dropoutRHS);
                        }
                    }
                }
                else
                {
                    results.AddRange(ex.RHSFeatures[r]); //TODO: implement dropout , args_->dropoutRHS);
                }
            }
        }

        private void GetRandomWord(ThreadState state, List<Base> result)
        {
            result.Add(state.WordNegatives[state.LastWordNegative]);
            state.LastWordNegative++;
            if (state.LastWordNegative >= state.WordNegatives.Length)
            {
                state.LastWordNegative = 0;
            }
        }

#if NETCOREAPP3_0
        private void ProjectRHS(SharedState state, List<Base> ws, Span<float> retval)
#else
        private void ProjectRHS(SharedState state, List<Base> ws, float[] retval)
#endif
        {
            Forward(state.RHSEmbeddings, ws, retval);
            if (ws.Count > 0)
            {
                var norm = (float)(Data.Similarity == SimilarityType.Dot ? Math.Pow(ws.Count, Data.P) : Norm2(retval));
                SIMD.Multiply(retval, 1f / norm);
            }
        }

#if NETCOREAPP3_0
        private void Forward(Matrix matrix, List<Base> ws, Span<float> retval)
        {
            retval.Fill(0f);
#else
        private void Forward(Matrix matrix, List<Base> ws, float[] retval)
        {
            retval.Zero();
#endif
            foreach (var b in ws)
            {
                var row = matrix.GetRow(b.ID);
                SIMD.Add(retval, row);
            }
        }


#if NETCOREAPP3_0
        private double Norm2(Span<float> a)
        {
            const float Epsilon = 1.192092896e-07F;
            var norm = (float)Math.Sqrt(SIMD.DotProduct(a, a));
            return (norm < Epsilon) ? Epsilon : norm;
        }
#else
        private double Norm2(float[] a)
        {
            const float Epsilon = 1.192092896e-07F;
            var norm = (float)Math.Sqrt(SIMD.DotProduct(a, a));
            return (norm < Epsilon) ? Epsilon : norm;
        }
#endif

#if NETCOREAPP3_0
        private void ProjectLHS(SharedState state, List<Base> ws, Span<float> retval)
#else
        private void ProjectLHS(SharedState state, List<Base> ws, float[] retval)
#endif
        {
            Forward(state.LHSEmbeddings, ws, retval);
            if (ws.Count > 0)
            {
                var norm = (float)(Data.Similarity == SimilarityType.Dot ? Math.Pow(ws.Count, Data.P) : Norm2(retval));
                SIMD.Multiply(retval, 1f / norm);
            }
        }

        private float TrainNLLBatch(ThreadState state, List<ParseResults> batch_exs, int negSearchLimit, float rate0, bool trainWord)
        {
            var batch_sz = batch_exs.Count;

            Span<float> posSim = stackalloc float[batch_sz];
            Span<float> labelRate = stackalloc float[batch_sz];
            Span<float> loss = stackalloc float[batch_sz];
            Span<int> num_negs = stackalloc int[batch_sz];

            float total_loss = 0f;
            var cols = Data.Dimensions;

            for (int i = 0; i < batch_sz; i++) { labelRate[i] = rate0; }

            for (int i = 0; i < batch_sz; i++)
            {
                ProjectLHS(state.Shared, batch_exs[i].LHSTokens, state.lhs[i]);
                ProjectRHS(state.Shared, batch_exs[i].RHSTokens, state.rhsP[i]);
            }

            state.batch_negLabels.Clear();

            for (int i = 0; i < negSearchLimit; i++)
            {
                var negLabels = new List<Base>();
                if (trainWord)
                {
                    GetRandomWord(state, negLabels);
                }
                else
                {
                    GetRandomRHS(state, negLabels);
                }
                ProjectRHS(state.Shared, negLabels, state.rhsN[i]);
                state.batch_negLabels.Add(negLabels);
            }

            var index = new List<int>();

            for (int i = 0; i < batch_sz; i++)
            {
                index.Clear();

                int cls_cnt = 1;
                state.prob[i].Clear();
                state.prob[i].Add(SIMD.DotProduct(state.lhs[i], state.rhsP[i]));
                float max = state.prob[i][0];

                for (int j = 0; j < negSearchLimit; j++)
                {
                    state.nRate[i][j] = 0f;
                    if (state.batch_negLabels[j] == batch_exs[i].RHSTokens) { continue; }
                    state.prob[i].Add(SIMD.DotProduct(state.lhs[i], state.rhsN[j]));
                    max = Math.Max(state.prob[i][0], state.prob[i][cls_cnt]);
                    index.Add(j);
                    cls_cnt += 1;
                }
                loss[i] = 0f;

                // skip, failed to find any negatives
                if (cls_cnt == 1) { continue; }

                num_negs[i] = cls_cnt - 1;
                float fbase = 0;
                for (int j = 0; j < cls_cnt; j++)
                {
                    state.prob[i][j] = (float)Math.Exp(state.prob[i][j] - max);
                    fbase += state.prob[i][j];
                }
                fbase = 1f / fbase;
                // normalize probabilities
                for (int j = 0; j < cls_cnt; j++)
                {
                    state.prob[i][j] *= fbase;
                }

                loss[i] = (float)-Math.Log(state.prob[i][0]);
                total_loss += loss[i];

                // Let w be the average of the words in the post, t+ be the
                // positive example (the tag the post has) and t- be the average
                // of the negative examples (the tags we searched for with submarginal
                // separation above).
                // Our error E is:
                //
                //    E = - log P(t+)
                //
                // Where P(t) = exp(dot(w, t)) / (\sum_{t'} exp(dot(w, t')))
                //
                // Differentiating term-by-term we get:
                //
                //    dE / dw = t+ (P(t+) - 1)
                //    dE / dt+ = w (P(t+) - 1)
                //    dE / dt- = w P(t-)

                state.rhsP[i].AsSpan().CopyTo(state.gradW[i].AsSpan());

                SIMD.Multiply(state.gradW[i], state.prob[i][0] - 1);

                for (int j = 1; j < cls_cnt; j++)
                {
                    var inj = index[j - 1];
                    SIMD.MultiplyAndAdd(state.gradW[i], state.rhsN[inj], state.prob[i][j]); //TODO Check if equivalent to this: gradW[i].add(rhsN[inj], prob[i][j]);
                    state.nRate[i][inj] = state.prob[i][j] * rate0;
                }

                labelRate[i] = (state.prob[i][0] - 1) * rate0;
            }

            Backward(state, batch_exs, state.batch_negLabels, ref state.gradW, ref state.lhs, num_negs, rate0, labelRate, ref state.nRate);

            return total_loss;
        }

        private float TripleLoss(float posSim, float negSim)
        {
            // We want the max representable loss to have some wiggle room to compute with.
            var val = Data.Margin - posSim + negSim;
            var kMaxLoss = 10e7;
            return (float)Math.Max(Math.Min(val, kMaxLoss), 0.0);
        }

        private float Similarity(ref float[] lhs, ref float[] rhs)
        {
            return Data.Similarity == SimilarityType.Dot ? SIMD.DotProduct(lhs, rhs) : SIMD.CosineSimilarity(lhs, rhs);
        }

        private ParseResults GetExample(ThreadState state, int i)
        {
            return Convert(state.Corpus[i]);
        }

        private ParseResults Convert(ParseResults example)
        {
            var result = new ParseResults();
            result.Weight = example.Weight;
            result.LHSTokens.AddRange(example.LHSTokens);

            switch (Data.Type)
            {
                case ModelType.GraphSpace:
                //case ModelType.ImageSpace:
                //case ModelType.TagSpace:
                {
                    // lhs is the same, pick one random label as rhs
                    result.RHSTokens.AddRange(example.RHSFeatures[ThreadSafeFastRandom.Next(example.RHSFeatures.Length)]);
                    break;
                }
                case ModelType.PageSpace:
                //case ModelType.DocSpace:
                {
                    // pick one random label as rhs and the rest is lhs
                    var idx = ThreadSafeFastRandom.Next(example.RHSFeatures.Length);
                    for (int i = 0; i < example.RHSFeatures.Length; i++)
                    {
                        if (i == idx)
                        {
                            result.RHSTokens.AddRange(example.RHSFeatures[i]);
                        }
                        else
                        {
                            result.LHSTokens.AddRange(example.RHSFeatures[i]);
                        }
                    }

                    break;
                }
                case ModelType.ArticleSpace:
                {
                    // pick one random label as lhs and the rest is rhs
                    var idx = ThreadSafeFastRandom.Next(example.RHSFeatures.Length);
                    for (int i = 0; i < example.RHSFeatures.Length; i++)
                    {
                        if (i == idx)
                        {
                            result.LHSTokens.AddRange(example.RHSFeatures[i]);
                        }
                        else
                        {
                            result.RHSTokens.AddRange(example.RHSFeatures[i]);
                        }
                    }

                    break;
                }
                case ModelType.SentenceSpace:
                {
                    // pick two random labels, one as lhs and the other as rhs
                    var idx = ThreadSafeFastRandom.Next(example.RHSFeatures.Length);
                    int idx2;
                    do
                    {
                        idx2 = ThreadSafeFastRandom.Next(example.RHSFeatures.Length);
                    } while (idx2 == idx);
                    result.LHSTokens.AddRange(example.RHSFeatures[idx]);
                    result.RHSTokens.AddRange(example.RHSFeatures[idx2]);
                    break;
                }
                case ModelType.RelationalSpace:
                {
                    // the first one as lhs and the second one as rhs
                    result.LHSTokens.AddRange(example.RHSFeatures[0]);
                    result.RHSTokens.AddRange(example.RHSFeatures[1]);
                    break;
                }
                case ModelType.WordEmbeddings:
                {
                    //TODO throw invalid operation ??
                    break;
                }
            }

            return result;
        }

        private void GetWordExample(ThreadState state, int i, List<ParseResults> results)
        {
            GetWordExamples(state.Corpus[i].LHSTokens, results);
        }

        private void GetWordExamples(List<Base> doc, List<ParseResults> results)
        {
            results.Clear();
            for (int widx = 0; widx < doc.Count; widx++)
            {
                var rslt = new ParseResults();
                rslt.RHSTokens.Add(doc[widx]);
                for (int i = Math.Max(widx - Data.ContextWindow, 0); i < Math.Min(widx + Data.ContextWindow, doc.Count); i++)
                {
                    if (i != widx)
                    {
                        rslt.LHSTokens.Add(doc[i]);
                    }
                }
                rslt.Weight = Data.WordWeight;
                results.Add(rslt);
            }
        }

        public uint HashToken(IToken tk, Language language)
        {
            return HashToken(tk.ValueAsSpan, tk.POS, language);
        }

        public uint HashToken(ReadOnlySpan<char> value, PartOfSpeech pos, Language language)
        {
            if (Data.InvariantCase)
            {
                return (uint)Hashes.CombineWeak(Hashes.CombineWeak(Hashes.IgnoreCaseHash32(value), POS_Hashes[(int)pos]), Language_Hashes[(int)language]);
            }
            else
            {
                return (uint)Hashes.CombineWeak(Hashes.CombineWeak(Hashes.CaseSensitiveHash32(value), POS_Hashes[(int)pos]), Language_Hashes[(int)language]);
            }
        }

        public struct Base
        {
            public int ID;
            public uint Hash;
            public float Weight;

            public Base(int id, uint hash, float weight)
            {
                ID = id;
                Hash = hash;
                Weight = weight;
            }
        }

        internal class ParseResults
        {
            public float Weight;
            public Base[][] RHSFeatures;
            public List<Base> LHSTokens;
            public List<Base> RHSTokens; //important to be a List, as we compare the reference

            public ParseResults()
            {
                LHSTokens = new List<Base>();
                RHSTokens = new List<Base>();
            }
        }

        internal class SharedState
        {
            public Matrix RHSEmbeddings;
            public Matrix LHSEmbeddings;
            public float[] RHSUpdates;
            public float[] LHSUpdates;

            public void InitializeModelWeights(int dimensions, int nWords, int nLabels, int nBuckets, bool shareEmbeddings, bool useAdaGrad, float stdDev)
            {
                int lhsSize = nWords + nLabels + nBuckets; //nBuckets is > 0 if we are training with ngrams

                LHSEmbeddings = new Matrix(lhsSize, dimensions);
                //TODO INIT MATRIX LIKE SPARELINEAR with initRandSd
                LHSEmbeddings.Uniform(stdDev); //TODO This should have been a normal distribution - check if worth it: https://stackoverflow.com/questions/2325472/generate-random-numbers-following-a-normal-distribution-in-c-c

                if (shareEmbeddings)
                {
                    RHSEmbeddings = LHSEmbeddings;
                }
                else
                {
                    RHSEmbeddings = new Matrix(lhsSize, dimensions);
                    RHSEmbeddings.Uniform(stdDev); //TODO This should have been a normal distribution
                }

                if (useAdaGrad)
                {
                    LHSUpdates = new float[LHSEmbeddings.Rows];
                    RHSUpdates = new float[RHSEmbeddings.Rows];
                }
            }
        }

        public class InputData
        {
            public ConcurrentDictionary<int, List<List<Base>>> Corpus = new ConcurrentDictionary<int, List<List<Base>>>();
            public ConcurrentDictionary<uint, SingleToken> uniqueIDs = new ConcurrentDictionary<uint, SingleToken>();
            public ConcurrentDictionary<uint, SingleToken> uniqueTokens = new ConcurrentDictionary<uint, SingleToken>();
            public int docCount;
        }

        internal class ThreadState
        {
            private const int MAX_VOCAB_SIZE = 10000000;
            private const int MAX_WORD_NEGATIVES_SIZE = 10000000;

            public int ThreadID;
            public float Loss;
            public long NumberOfExamples;
            public CancellationToken CancellationToken;

            public int NegativePosition;

            public ParseResults[] Corpus;
            public int Counts;
            public float Rate;
            internal float FinishRate;
            internal int Epoch;

            public Base[] WordNegatives;

            public SharedState Shared;
            internal int LastWordNegative;

            public Measure Measure;
            internal List<float>[] prob;
            internal List<List<Base>> batch_negLabels;
            internal float[][] gradW;
            internal float[][] lhs;
            internal float[][] rhsP;
            internal float[][] rhsN;
            internal float[][] nRate;

            public TrainingHistory TrainingHistory { get; internal set; }

            public ThreadState(ParseResults[] corpus, int thread, CancellationToken token, SharedState sharedState, Measure measure, int batch_sz, int negSearchLimit, int dim)
            {
                Loss = 0f;
                NumberOfExamples = 1;
                NegativePosition = 0;
                ThreadID = thread;
                CancellationToken = token;
                Corpus = corpus;
                Shared = sharedState;
                Measure = measure;
                prob = Enumerable.Range(0, batch_sz).Select(_ => new List<float>()).ToArray();
                batch_negLabels = new List<List<Base>>();

                gradW = Enumerable.Range(0, batch_sz).Select(_ => new float[dim]).ToArray(); //new float[batch_sz][];
                lhs = Enumerable.Range(0, batch_sz).Select(_ => new float[dim]).ToArray(); //new float[batch_sz][];
                rhsP = Enumerable.Range(0, batch_sz).Select(_ => new float[dim]).ToArray(); // new float[batch_sz][];
                rhsN = Enumerable.Range(0, negSearchLimit).Select(_ => new float[dim]).ToArray(); //new float[negSearchLimit][];
                nRate = Enumerable.Range(0, batch_sz).Select(_ => new float[negSearchLimit]).ToArray();
            }

            internal void InitializeWordNegatives()
            {
                WordNegatives = new Base[MAX_WORD_NEGATIVES_SIZE];
                for (int i = 0; i < MAX_WORD_NEGATIVES_SIZE; i++)
                {
                    WordNegatives[i] = GetRandomWord();
                }
            }

            private Base GetRandomWord()
            {
                var ex = Corpus[ThreadSafeFastRandom.Next(Corpus.Length)];
                return ex.LHSTokens[ThreadSafeFastRandom.Next(ex.LHSTokens.Count)];
            }
        }

        public enum ModelType
        {
            TagSpace = 0,
            GraphSpace = 0, //input is tuples of (node -> edge -> node)
            ImageSpace = 0, //Need to use weight for each feature, and id = [0..embedding dimensions]  - i.e. Base[11] = (11, -0.9231)
            PageSpace = 1,
            DocSpace = 1,
            ArticleSpace = 2,
            SentenceSpace = 3,
            RelationalSpace = 4,
            WordEmbeddings = 5
        }

        public enum LossType
        {
            SoftMax,
            Hinge
        }

        public enum SimilarityType
        {
            Dot,
            Cosine
        }

        public enum InputType
        {
            FastText,
            LabeledDocuments
        }
    }
}