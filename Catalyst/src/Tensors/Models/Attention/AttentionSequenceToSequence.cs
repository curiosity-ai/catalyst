using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Catalyst.Tensors;
using Catalyst.Tensors.CUDA;
using Mosaik.Core;
using Microsoft.Extensions.Logging;
using Catalyst.Models;

namespace Catalyst.Tensors.Models
{
    public class AttentionSequenceToSequenceData : StorableObjectData
    {
        public float ClipGradientsValue { get; set; } = 5.0f;
        public int Depth { get; set; } = 1;
        public int HiddenSize { get; set; } = 128;
        public int WordVectorSize { get; set; } = 128;
        public float StartLearningRate { get; set; } = 0.001f;
        public float L2RegularizationStrength { get; set; } = 0.000001f;
        public float DropoutRatio { get; set; } = 0.1f;

        public int MaximumGeneratedSentenceLength = 100;
        public int LetterSize { get; set; }

        public int BatchSize { get; set; } = 1;

        public ConcurrentDictionary<string, int> SourceWordToIndex;
        public ConcurrentDictionary<int, string> IndexToSourceWord;

        public ConcurrentDictionary<string, int> TargetWordToIndex;
        public ConcurrentDictionary<int, string> IndexToTargetWord;

        public ArchTypeEnums ArchType { get; set;  }  = ArchTypeEnums.CPU;
        public bool UseDropout { get; set; } = false;
        public int Epochs { get; set; } = 5;
        public int MinimumWordCount { get; set; } = 5;
    }

    public class AttentionSequenceToSequence : StorableObject<AttentionSequenceToSequence, AttentionSequenceToSequenceData>
    {
        public static int[] DeviceIDs = new int[0];

        public event EventHandler<TrainingEvent> IterationDone;
        
        private const string m_UNK = "<UNK>";
        private const string m_END = "<END>";
        private const string m_START = "<START>";
        private IWeightFactory[] m_weightFactory;
        private int m_maxWord = 100;
        private Optimizer Solver;

        private IWeightMatrix[] SourceEmbeddings;
        private IWeightMatrix[] TargetEmbeddings;
        private BiEncoder[] BiEncoder;
        private AttentionDecoder[] Decoder;
        private FeedForwardLayer[] DecoderFeedForwardLayer;
        
        private int DefaultDeviceID_Decoder = 0;
        private int DefaultDeviceID_SourceEmbeddings = 0;
        private int DefaultDeviceID_TargetEmbeddings = 0;
        private int DefaultDeviceID_BiEncoder = 0;
        private int DefaultDeviceID_DecoderFeedForwardLayer = 0;

        // optimization  hyperparameters
        
        private int m_defaultDeviceId = 0;

        public new static async Task<AttentionSequenceToSequence> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new AttentionSequenceToSequence(language, version, tag);
            await a.LoadDataAsync();

            var s1 = await DataStore.OpenReadAsync(a.Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(Models.BiEncoder), a.Version, a.Tag + "-" + nameof(BiEncoder));
            var s2 = await DataStore.OpenReadAsync(a.Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(IWeightMatrix),    a.Version, a.Tag + "-" + nameof(SourceEmbeddings));
            var s3 = await DataStore.OpenReadAsync(a.Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(IWeightMatrix),    a.Version, a.Tag + "-" + nameof(TargetEmbeddings));
            var s4 = await DataStore.OpenReadAsync(a.Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(AttentionDecoder), a.Version, a.Tag + "-" + nameof(Decoder));
            var s5 = await DataStore.OpenReadAsync(a.Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(FeedForwardLayer), a.Version, a.Tag + "-" + nameof(DecoderFeedForwardLayer));

            a.Initialize();

            a.BiEncoder[a.DefaultDeviceID_BiEncoder].Load(s1);
            a.SourceEmbeddings[a.DefaultDeviceID_SourceEmbeddings].Load(s2);
            a.TargetEmbeddings[a.DefaultDeviceID_TargetEmbeddings].Load(s3);
            a.Decoder[a.DefaultDeviceID_Decoder].Load(s4);
            a.DecoderFeedForwardLayer[a.DefaultDeviceID_DecoderFeedForwardLayer].Load(s5);

            return a;
        }

        public override async Task StoreAsync()
        {
            var s1 = await DataStore.OpenWriteAsync(Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(Models.BiEncoder),  Version, Tag +  "-" + nameof(BiEncoder));
            var s2 = await DataStore.OpenWriteAsync(Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(IWeightMatrix),     Version, Tag +  "-" + nameof(SourceEmbeddings));
            var s3 = await DataStore.OpenWriteAsync(Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(IWeightMatrix),     Version, Tag +  "-" + nameof(TargetEmbeddings));
            var s4 = await DataStore.OpenWriteAsync(Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(AttentionDecoder),  Version, Tag +  "-" + nameof(Decoder));
            var s5 = await DataStore.OpenWriteAsync(Language, nameof(AttentionSequenceToSequenceData) + "-" + nameof(FeedForwardLayer),  Version, Tag +  "-" + nameof(DecoderFeedForwardLayer));

            s1.SetLength(0);
            s2.SetLength(0);
            s3.SetLength(0);
            s4.SetLength(0);
            s5.SetLength(0);

            BiEncoder[DefaultDeviceID_BiEncoder].Save(s1);
            SourceEmbeddings[DefaultDeviceID_SourceEmbeddings].Save(s2);
            TargetEmbeddings[DefaultDeviceID_TargetEmbeddings].Save(s3);
            Decoder[DefaultDeviceID_Decoder].Save(s4);
            DecoderFeedForwardLayer[DefaultDeviceID_DecoderFeedForwardLayer].Save(s5);

            s1.Close();
            s2.Close();
            s3.Close();
            s4.Close();
            s5.Close();

            await base.StoreAsync();
        }
        bool initialized = false;

        private void Initialize()
        {
            if(initialized) { return; }
            initialized = true;
            CheckBatchSizeAndDeviceIDs();

            InitWeights();

            if (Data.ArchType == ArchTypeEnums.GPU_CUDA)
            {
                TensorAllocator.InitDevices(DeviceIDs);
                SetDefaultDeviceIds(DeviceIDs.Length);
            }

            InitWeightsFactory();
            SetBatchSize(Data.BatchSize);
        }

        public AttentionSequenceToSequence(Language language, int version, string tag) : base(language, version, tag)
        {

        }

        public void Train(Corpus trainCorpus)
        {
            CheckBatchSizeAndDeviceIDs();

            if (Data.ArchType == ArchTypeEnums.GPU_CUDA)
            {
                TensorAllocator.InitDevices(DeviceIDs);
                SetDefaultDeviceIds(DeviceIDs.Length);
            }


            Initialize();

            Logger.LogInformation("Start to train...");

            Solver = new Optimizer();

            AccumulatedCostInLastEpoch = double.MaxValue;

            float learningRate = Data.StartLearningRate;
            for (int i = 0; i < Data.Epochs; i++)
            {
                TrainEp(i, learningRate, trainCorpus);
                learningRate = Data.StartLearningRate / (1.0f + 0.95f * (i + 1));
            }
        }

        private void SetDefaultDeviceIds(int deviceNum)
        {
            int i = 0;

            DefaultDeviceID_SourceEmbeddings = (i++) % deviceNum;
            DefaultDeviceID_TargetEmbeddings = (i++) % deviceNum;

            DefaultDeviceID_BiEncoder = (i++) % deviceNum;
            DefaultDeviceID_Decoder = (i++) % deviceNum;

            DefaultDeviceID_DecoderFeedForwardLayer = (i++) % deviceNum;
        }

        private void CheckBatchSizeAndDeviceIDs()
        {
            if (Data.ArchType != ArchTypeEnums.GPU_CUDA)
            {
                Data.BatchSize = 1;
                DeviceIDs = Enumerable.Range(0, Environment.ProcessorCount).ToArray();
            }
        }

        private void SetBatchSize(int batchSize)
        {
            Data.BatchSize = batchSize;

            for (int i = 0; i < DeviceIDs.Length; i++)
            {

                if (BiEncoder[i] != null)
                {
                    BiEncoder[i].SetBatchSize(m_weightFactory[i], batchSize);
                }

                if (Decoder[i] != null)
                {
                    Decoder[i].SetBatchSize(m_weightFactory[i], batchSize);
                }
            }
        }

        private void InitWeightsFactory()
        {
            m_weightFactory = new IWeightFactory[DeviceIDs.Length];
            if (Data.ArchType == ArchTypeEnums.GPU_CUDA)
            {
                for (int i = 0; i < DeviceIDs.Length; i++)
                {
                    m_weightFactory[i] = new WeightTensorFactory();
                }
            }
            else
            {
                for (int i = 0; i < DeviceIDs.Length; i++)
                {
                    m_weightFactory[i] = new WeightMatrixFactory();
                }
            }
        }

        private void InitWeights()
        {
            Logger.LogInformation($"Initializing weights...");

            SourceEmbeddings = new IWeightMatrix[DeviceIDs.Length];
            TargetEmbeddings = new IWeightMatrix[DeviceIDs.Length];

            BiEncoder = new BiEncoder[DeviceIDs.Length];
            Decoder = new AttentionDecoder[DeviceIDs.Length];

            DecoderFeedForwardLayer = new FeedForwardLayer[DeviceIDs.Length];

            for (int i = 0; i < DeviceIDs.Length; i++)
            {
                Logger.LogInformation($"Initializing weights for device '{DeviceIDs[i]}'");
                if (Data.ArchType == ArchTypeEnums.GPU_CUDA)
                {
                    SourceEmbeddings[i] = new WeightTensor(Data.IndexToSourceWord.Count, Data.WordVectorSize, DeviceIDs[i], true);
                    TargetEmbeddings[i] = new WeightTensor(Data.IndexToTargetWord.Count + 3, Data.WordVectorSize, DeviceIDs[i], true);
                }
                else
                {
                    SourceEmbeddings[i] = new WeightMatrix(Data.IndexToSourceWord.Count, Data.WordVectorSize, true);
                    TargetEmbeddings[i] = new WeightMatrix(Data.IndexToTargetWord.Count + 3, Data.WordVectorSize, true);
                }

                Logger.LogInformation($"Initializing encoders and decoders for device '{DeviceIDs[i]}'...");

                BiEncoder[i] = new BiEncoder(Data.BatchSize, Data.HiddenSize, Data.WordVectorSize, Data.Depth, Data.ArchType, DeviceIDs[i]);
                Decoder[i] = new AttentionDecoder(Data.BatchSize, Data.HiddenSize, Data.WordVectorSize, Data.HiddenSize * 2, Data.Depth, Data.ArchType, DeviceIDs[i]);

                DecoderFeedForwardLayer[i] = new FeedForwardLayer(Data.HiddenSize, Data.IndexToTargetWord.Count + 3, Data.ArchType, DeviceIDs[i]);
            }

            InitWeightsFactory();
        }


        public void UseFastTextEmbeddings(FastText sourceModel, FastText targetModel)
        {
            Initialize();

            for (int i = 0; i < DeviceIDs.Length; i++)
            {
                //If pre-trained embedding weights are speicifed, loading them from files
                if (sourceModel is object)
                {
                    Logger.LogInformation($"Loading ExtEmbedding model from '{sourceModel.GetStoredObjectInfo().ToString()}' for source side.");
                    LoadFastTextEmbeddings(sourceModel, SourceEmbeddings[i], Data.SourceWordToIndex);
                }

                if (targetModel is object)
                {
                    Logger.LogInformation($"Loading ExtEmbedding model from '{targetModel.GetStoredObjectInfo().ToString()}' for target side.");
                    LoadFastTextEmbeddings(targetModel, TargetEmbeddings[i], Data.TargetWordToIndex);
                }
            }
        }

        private void LoadFastTextEmbeddings(FastText model, IWeightMatrix embeddingMatrix, ConcurrentDictionary<string, int> wordToIndex)
        {
            if (model.Data.Dimensions != embeddingMatrix.Columns)
            {
                throw new ArgumentException($"Inconsistent embedding size. ExtEmbeddingModel size = '{model.Data.Dimensions}', EmbeddingMatrix column size = '{embeddingMatrix.Columns}'");
            }

            foreach (KeyValuePair<string, int> pair in wordToIndex)
            {
                float[] vector = model.GetVector(pair.Key, Language, model.GetMostProbablePOSforWord(pair.Key));

                if (vector != null)
                {
                    embeddingMatrix.SetWeightAtRow(pair.Value, vector);
                }
            }
        }

        public void InitializeVocabulary(Corpus trainCorpus)
        {
            Logger.LogInformation("Building vocabulary from training corpus...");
            
            var sourceW2I = new Dictionary<string, int>();
            var targetW2I = new Dictionary<string, int>();

            Data.SourceWordToIndex = new ConcurrentDictionary<string, int>();
            Data.IndexToSourceWord = new ConcurrentDictionary<int, string>();
            
            Data.TargetWordToIndex = new ConcurrentDictionary<string, int>();
            Data.IndexToTargetWord = new ConcurrentDictionary<int, string>();

            foreach (var pair in trainCorpus)
            {
                foreach(var txti in pair.Source)
                {
                    var val = sourceW2I.TryGetValue(txti, out var v) ? v : 0;
                    sourceW2I[txti] = val + 1;
                }

                foreach(var txti in pair.Target)
                {
                    var val = targetW2I.TryGetValue(txti, out var v) ? v : 0;
                    targetW2I[txti] = val + 1;
                }
            }

            Data.SourceWordToIndex[m_END] = (int)SENTTAGS.END;
            Data.SourceWordToIndex[m_START] = (int)SENTTAGS.START;
            Data.SourceWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            Data.IndexToSourceWord[(int)SENTTAGS.END] = m_END;
            Data.IndexToSourceWord[(int)SENTTAGS.START] = m_START;
            Data.IndexToSourceWord[(int)SENTTAGS.UNK] = m_UNK;

            Data.TargetWordToIndex[m_END] = (int)SENTTAGS.END;
            Data.TargetWordToIndex[m_START] = (int)SENTTAGS.START;
            Data.TargetWordToIndex[m_UNK] = (int)SENTTAGS.UNK;

            Data.IndexToTargetWord[(int)SENTTAGS.END] = m_END;
            Data.IndexToTargetWord[(int)SENTTAGS.START] = m_START;
            Data.IndexToTargetWord[(int)SENTTAGS.UNK] = m_UNK;

            var k = 3;
            foreach (var ch in sourceW2I)
            {
                if (ch.Value >= Data.MinimumWordCount)
                {
                    // add word to vocab
                    Data.SourceWordToIndex[ch.Key] = k;
                    Data.IndexToSourceWord[k] = ch.Key;
                    k++;
                }
            }

            Logger.LogInformation($"Source Vocabulary Size = '{k}'");

            k = 3;
            foreach (var ch in targetW2I)
            {
                if (ch.Value >= Data.MinimumWordCount)
                {
                    // add word to vocab
                    Data.TargetWordToIndex[ch.Key] = k;
                    Data.IndexToTargetWord[k] = ch.Key;
                    k++;
                }
            }

            Logger.LogInformation($"Target Vocabulary Size = '{k}'");
        }

        private object locker = new object();

        private double AccumulatedCostInLastEpoch = double.MaxValue;

        private void TrainEp(int ep, float learningRate, Corpus trainCorpus)
        {
            int processedPairsCount = 0;
            int parameterUpdateCount = 0;

            DateTimeOffset startDateTime = DateTime.UtcNow;

            double accumulatedCostInCurrentEpoch = 0.0;
            long sourceWordCount = 0;
            long targetWordCount = 0;

            int totalBatchSize = Data.BatchSize * DeviceIDs.Length;

            List<TokenPairs> sentencePairs = new List<TokenPairs>();

            TensorAllocator.FreeMemoryAllDevices();

            Logger.LogInformation($"Base learning rate is '{learningRate}' at epoch '{ep}'");

            //Clean caches of parameter optmization
            Logger.LogInformation($"Cleaning cache of weights optmiazation.'");
            CleanWeightCache();

            Logger.LogInformation($"Start to process training corpus.");
            
            foreach (var pair in trainCorpus)
            {
                sentencePairs.Add(pair);
                
                if (sentencePairs.Count == totalBatchSize)
                {
                    ProcessBatch(sentencePairs);
                    sentencePairs.Clear();
                }
            }

            if (sentencePairs.Count > 0)
            {
                ProcessBatch(sentencePairs);
                sentencePairs.Clear();
            }

            Logger.LogInformation($"Epoch '{ep}' took '{DateTime.Now - startDateTime}' time to finish. AvgCost = {accumulatedCostInCurrentEpoch/ targetWordCount:n6}");

            if (AccumulatedCostInLastEpoch > accumulatedCostInCurrentEpoch)
            {
                StoreAsync().Wait();
                TensorAllocator.FreeMemoryAllDevices();
                AccumulatedCostInLastEpoch = accumulatedCostInCurrentEpoch;
            }

            void ProcessBatch(List<TokenPairs> pairs)
            {
                float cost = 0.0f;
                var targetWordCountBefore = targetWordCount;
                Reset();
                
                SyncWeights(); //Copy weights from weights kept in default device to all other devices
                int threadID = 0; 
                Parallel.ForEach(pairs.SplitIntoN(DeviceIDs.Length), batchPairs =>
                {
                    var thID = Interlocked.Increment(ref threadID) % DeviceIDs.Length;
                    var srcSnts = new List<List<string>>();
                    var tgtSnts = new List<List<string>>();

                    var slen = 0;
                    var tlen = 0;
                    foreach (var batchPair in batchPairs)
                    {
                        var srcSnt = new List<string>();

                        //Add BOS and EOS tags to source sentences
                        srcSnt.Add(m_START);
                        srcSnt.AddRange(batchPair.Source);
                        srcSnt.Add(m_END);

                        srcSnts.Add(srcSnt);
                        tgtSnts.Add(batchPair.Target.ToList());

                        slen += srcSnt.Count;
                        tlen += batchPair.Target.Length;
                    }

                    Interlocked.Add(ref sourceWordCount, slen);
                    Interlocked.Add(ref targetWordCount, tlen);

                    var computeGraph = CreateComputGraph(thID);

                    var encodedWeightMatrix = Encode(computeGraph, srcSnts, BiEncoder[thID], SourceEmbeddings[thID]);
                        
                    float lcost = Decode(tgtSnts, computeGraph, encodedWeightMatrix, Decoder[thID], DecoderFeedForwardLayer[thID], TargetEmbeddings[thID], out _);

                    lock (locker)
                    {
                        cost += lcost;
                    }

                    computeGraph.Backward();
                });

                //Sum up gradients in all devices, and kept it in default device for parameters optmization
                SyncGradientsBackToDefaultDevices();

                if (float.IsInfinity(cost) == false && float.IsNaN(cost) == false)
                {
                    processedPairsCount += pairs.Count;
                    accumulatedCostInCurrentEpoch += cost;
                }
                else
                {
                    Logger.LogInformation($"Invalid cost value.");
                }

                float avgAllLR = UpdateParameters(learningRate, Data.BatchSize);
                parameterUpdateCount++;

                ClearGradient();

                if (IterationDone != null /*&& processedPairsCount % (100 * Data.BatchSize) == 0*/)
                {
                    IterationDone(this, new TrainingEvent()
                    {
                        LearningRate = avgAllLR,
                        Loss = cost / (targetWordCount - targetWordCountBefore),
                        Epoch = ep,
                        Update = parameterUpdateCount,
                        SpansCount = processedPairsCount,
                        TokensCount = sourceWordCount * 2 + targetWordCount,
                        StartDateTime = startDateTime
                    });
                }
            }
        }

        private IComputeGraph CreateComputGraph(int deviceIdIdx, bool needBack = true)
        {
            IComputeGraph g;
            if (Data.ArchType == ArchTypeEnums.CPU_MKL)
            {
                g = new ComputeGraphMKL(m_weightFactory[deviceIdIdx], needBack);
            }
            else if (Data.ArchType == ArchTypeEnums.GPU_CUDA)
            {
                g = new ComputeGraphTensor(m_weightFactory[deviceIdIdx], DeviceIDs[deviceIdIdx], needBack);
            }
            else
            {
                g = new ComputeGraph(m_weightFactory[deviceIdIdx], needBack);
            }

            return g;
        }

        private List<int> PadSentences(List<List<string>> sentencesList)
        {
            List<int> originalLengths = new List<int>();

            int maxLen = sentencesList.Max(l =>l.Count);

            foreach (var l in sentencesList)
            {
                int count = l.Count;
                originalLengths.Add(count);

                for (int j = 0; j < maxLen - count; j++)
                {
                    l.Add(m_END);
                }
            }

            return originalLengths;
        }

        private IWeightMatrix Encode(IComputeGraph g, List<List<string>> inputSentences, BiEncoder biEncoder, IWeightMatrix Embedding)
        {
            PadSentences(inputSentences);

            int seqLen = inputSentences[0].Count;
            var forwardInput = new List<IWeightMatrix>();
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < inputSentences.Count; j++)
                {
                    var inputSentence = inputSentences[j];
                    int ix_source = (int)SENTTAGS.UNK;
                    if (Data.SourceWordToIndex.ContainsKey(inputSentence[i]))
                    {
                        ix_source = Data.SourceWordToIndex[inputSentence[i]];
                    }
                    else
                    {
                        Logger.LogTrace("{WORD} is an unknown word.", inputSentence[i]);
                    }
                    var x = g.PeekRow(Embedding, ix_source);
                    forwardInput.Add(x);
                }
            }

            var forwardInputsM = g.ConcatRows(forwardInput);
            List<IWeightMatrix> attResults = new List<IWeightMatrix>();
            for (int i = 0; i < seqLen; i++)
            {
                var emb_i = g.PeekRow(forwardInputsM, i * inputSentences.Count, inputSentences.Count);             
                attResults.Add(emb_i);
            }

            var encodedOutput = biEncoder.Encode(attResults, g);
            var encodedOutput2 = g.ConcatRows(encodedOutput);

            return encodedOutput2;
        }

        private float Decode(List<List<string>> outputSentences, IComputeGraph g, IWeightMatrix encodedOutputs, AttentionDecoder decoder, FeedForwardLayer decoderFFLayer, IWeightMatrix Embedding, out List<List<string>> predictSentence)
        {
            predictSentence = null;
            float cost = 0.0f;
            var attPreProcessResult = decoder.PreProcess(encodedOutputs, g);

            var originalOutputLengths = PadSentences(outputSentences);
            int seqLen = outputSentences[0].Count;

            int[] ix_inputs = new int[Data.BatchSize];
            int[] ix_targets = new int[Data.BatchSize];
            for (int i = 0; i < ix_inputs.Length; i++)
            {
                ix_inputs[i] = (int)SENTTAGS.START;
            }

            for (int i = 0; i < seqLen + 1; i++)
            {
                //Get embedding for all sentence in the batch at position i
                List<IWeightMatrix> inputs = new List<IWeightMatrix>();
                for (int j = 0; j < Data.BatchSize; j++)
                {
                    List<string> OutputSentence = outputSentences[j];

                    ix_targets[j] = (int)SENTTAGS.UNK;
                    if (i >= seqLen)
                    {
                        ix_targets[j] = (int)SENTTAGS.END;
                    }
                    else
                    {
                        if (Data.TargetWordToIndex.ContainsKey(OutputSentence[i]))
                        {
                            ix_targets[j] = Data.TargetWordToIndex[OutputSentence[i]];
                        }
                    }

                    var x = g.PeekRow(Embedding, ix_inputs[j]);

                    inputs.Add(x);
                }
           
                var inputsM = g.ConcatRows(inputs);

                //Decode output sentence at position i
                var eOutput = decoder.Decode(inputsM, attPreProcessResult, g);
                if (Data.DropoutRatio > 0.0f)
                {
                    eOutput = g.Dropout(eOutput, Data.DropoutRatio);
                }

                var o = decoderFFLayer.Process(eOutput, g);

                //Softmax for output
//                var o = g.MulAdd(eOutput, Whd, bds);
                var probs = g.Softmax(o, false);

                o.ReleaseWeight();

                //Calculate loss for each word in the batch
                List<IWeightMatrix> probs_g = g.UnFolderRow(probs, Data.BatchSize, false);
                for (int k = 0; k < Data.BatchSize; k++)    
                {
                    var probs_k = probs_g[k];
                    var score_k = probs_k.GetWeightAt(ix_targets[k]);

                    if (i < originalOutputLengths[k] + 1)
                    {
                        cost += (float)-Math.Log(score_k);
                    }

                    probs_k.SetWeightAt(score_k - 1, ix_targets[k]);

                    ix_inputs[k] = ix_targets[k];
                    probs_k.Dispose();
                }

                o.SetGradientByWeight(probs);

                //Hacky: Run backward for last feed forward layer and dropout layer in order to save memory usage, since it's not time sequence dependency
                g.RunTopBackward();
                g.RunTopBackward();
                if (Data.DropoutRatio > 0.0f)
                {
                    g.RunTopBackward();
                }
            }

            return cost;
        }

        private float UpdateParameters(float learningRate, int batchSize)
        {
            var models = GetParametersFromDefaultDevice();
            return Solver.UpdateWeights(models, batchSize, learningRate, Data.L2RegularizationStrength, Data.ClipGradientsValue, Data.ArchType);
        }
    
        private List<IWeightMatrix> GetParametersFromDeviceAt(int i)
        {
            var model_i = BiEncoder[i].getParams();
            model_i.AddRange(Decoder[i].getParams());
            model_i.Add(SourceEmbeddings[i]);
            model_i.Add(TargetEmbeddings[i]);

            model_i.AddRange(DecoderFeedForwardLayer[i].getParams());

            return model_i;
        }

        private List<IWeightMatrix> GetParametersFromDefaultDevice()
        {
            var model = BiEncoder[DefaultDeviceID_BiEncoder].getParams();
            model.AddRange(Decoder[DefaultDeviceID_Decoder].getParams());
            model.Add(SourceEmbeddings[DefaultDeviceID_SourceEmbeddings]);
            model.Add(TargetEmbeddings[DefaultDeviceID_TargetEmbeddings]);

            model.AddRange(DecoderFeedForwardLayer[DefaultDeviceID_DecoderFeedForwardLayer].getParams());

            return model;
        }

        /// <summary>
        /// Copy weights in default device to all other devices
        /// </summary>
        private void SyncWeights()
        {
            var model = GetParametersFromDefaultDevice();           
            Parallel.For(0, DeviceIDs.Length, i =>
            {
                var model_i = GetParametersFromDeviceAt(i);
                for (int j = 0; j < model.Count; j++)
                {
                    if (model_i[j] != model[j])
                    {
                        model_i[j].CopyWeights(model[j]);
                    }
                }
            });         
        }

        private void ClearGradient()
        {
            Parallel.For(0, DeviceIDs.Length, i =>
            {
                var model_i = GetParametersFromDeviceAt(i);
                for (int j = 0; j < model_i.Count; j++)
                {
                    model_i[j].ClearGradient();
                }
            });
        }

        private void SyncGradientsBackToDefaultDevices()
        {
            var model = GetParametersFromDefaultDevice();
            
            foreach(var i in DeviceIDs)
            {
                var model_i = GetParametersFromDeviceAt(i);
                for (int j = 0; j < model.Count; j++)
                {
                    if (model[j] != model_i[j])
                    {
                        model[j].AddGradient(model_i[j]);
                    }
                }
            }
        }

        private void CleanWeightCache()
        {
            var model = GetParametersFromDefaultDevice();
            Solver.CleanCache(model);
        }

        private void Reset()
        {
            for (int i = 0; i < DeviceIDs.Length; i++)
            {
                m_weightFactory[i].Clear();

                BiEncoder[i].Reset(m_weightFactory[i]);
                Decoder[i].Reset(m_weightFactory[i]);
            }
        }

        public List<List<string>> Predict(List<string> input, int beamSearchSize = 1)
        {
            var biEncoder = BiEncoder[m_defaultDeviceId];
            var srcEmbedding = SourceEmbeddings[m_defaultDeviceId];
            var tgtEmbedding = TargetEmbeddings[m_defaultDeviceId];
            var decoder = Decoder[m_defaultDeviceId];
            var decoderFFLayer = DecoderFeedForwardLayer[m_defaultDeviceId];

            List<BeamSearchStatus> bssList = new List<BeamSearchStatus>();

            var g = CreateComputGraph(m_defaultDeviceId, false);
            Reset();

            List<string> inputSeq = new List<string>();
            inputSeq.Add(m_START);
            inputSeq.AddRange(input);
            inputSeq.Add(m_END);
         
            var inputSeqs = new List<List<string>>();
            inputSeqs.Add(inputSeq);
            IWeightMatrix encodedWeightMatrix = Encode(g, inputSeqs, biEncoder, srcEmbedding);

            var attPreProcessResult = decoder.PreProcess(encodedWeightMatrix, g);

            BeamSearchStatus bss = new BeamSearchStatus();
            bss.OutputIds.Add((int)SENTTAGS.START);
            bss.CTs = decoder.GetCTs();
            bss.HTs = decoder.GetHTs();

            bssList.Add(bss);

            List<BeamSearchStatus> newBSSList = new List<BeamSearchStatus>();
            bool finished = false;
            while (finished == false)
            {
                finished = true;
                for (int i = 0; i < bssList.Count; i++)
                {
                    bss = bssList[i];
                    if (bss.OutputIds[bss.OutputIds.Count - 1] == (int)SENTTAGS.END || bss.OutputIds.Count > m_maxWord)
                    {
                        newBSSList.Add(bss);
                    }
                    else
                    {
                        finished = false;
                        var ix_input = bss.OutputIds[bss.OutputIds.Count - 1];
                        decoder.SetCTs(bss.CTs);
                        decoder.SetHTs(bss.HTs);

                        var x = g.PeekRow(tgtEmbedding, ix_input);
                        var eOutput = decoder.Decode(x, attPreProcessResult, g);
                        var o = decoderFFLayer.Process(eOutput, g);

                        var probs = g.Softmax(o, false);

                        var preds = probs.GetTopNMaxWeightIdx(beamSearchSize);

                        for (int j = 0; j < preds.Count; j++)
                        {
                            BeamSearchStatus newBSS = new BeamSearchStatus();
                            newBSS.OutputIds.AddRange(bss.OutputIds);
                            newBSS.OutputIds.Add(preds[j]);

                            newBSS.CTs = decoder.GetCTs();
                            newBSS.HTs = decoder.GetHTs();

                            var score = probs.GetWeightAt(preds[j]);
                            newBSS.Score = bss.Score;
                            newBSS.Score += (float)(-Math.Log(score));

                            //var lengthPenalty = Math.Pow((5.0f + newBSS.OutputIds.Count) / 6, 0.6);
                            //newBSS.Score /= (float)lengthPenalty;

                            newBSSList.Add(newBSS);
                        }
                    }
                }

                bssList = GetTopNBSS(newBSSList, beamSearchSize);
                newBSSList.Clear();
            }
           
            List<List<string>> results = new List<List<string>>();
            for (int i = 0; i < bssList.Count; i++)
            {
                results.Add(PrintString(bssList[i].OutputIds));                
            }

            return results;
        }

        private List<string> PrintString(List<int> idxs)
        {
            List<string> result = new List<string>();
            foreach (var idx in idxs)
            {
                var letter = m_UNK;
                if (Data.IndexToTargetWord.ContainsKey(idx))
                {
                    letter = Data.IndexToTargetWord[idx];
                }
                result.Add(letter);
            }

            return result;
        }

        private List<BeamSearchStatus> GetTopNBSS(List<BeamSearchStatus> bssList, int topN)
        {
            FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>> q = new FixedSizePriorityQueue<ComparableItem<BeamSearchStatus>>(topN, new ComparableItemComparer<BeamSearchStatus>(false));

            for (int i = 0; i < bssList.Count; i++)
            {
                q.Enqueue(new ComparableItem<BeamSearchStatus>(bssList[i].Score, bssList[i]));
            }

            return q.Select(x => x.Value).ToList();         
        }
    }

    public enum SENTTAGS
    {
        END = 0,
        START,
        UNK
    }

    public class BeamSearchStatus
    {
        public List<int> OutputIds;
        public float Score;

        public List<IWeightMatrix> HTs;
        public List<IWeightMatrix> CTs;

        public BeamSearchStatus()
        {
            OutputIds = new List<int>();
            HTs = new List<IWeightMatrix>();
            CTs = new List<IWeightMatrix>();

            Score = 1.0f;
        }
    }
}
