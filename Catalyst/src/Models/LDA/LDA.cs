using UID;
using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using System.Globalization;
using System.IO;
using System.Threading;
using Microsoft.ML.Runtime;
using System.Security;
using System.Runtime.InteropServices;
using System.Collections.Immutable;
using Catalyst.Models.Native;
using System.Diagnostics;

namespace Catalyst.Models
{
    public class LDAModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public int NumberOfTopics               { get; set; } = 100;
        public float AlphaSum                   { get; set; } = 100; //Dirichlet prior on document-topic vectors
        public float Beta                       { get; set; } = 0.01f; //Dirichlet prior on vocab-topic vectors
        public int SamplingStepCount            { get; set; } = 4; //Number of Metropolis Hasting step
        public int MaximumNumberOfIterations    { get; set; } = 200;

        public int NumberOfThreads              { get; set; } = 1; 
        public int LikelihoodInterval           { get; set; } = 5; //Compute log likelihood over local dataset on this iteration interval
        public int MaximumTokenCountPerDocument { get; set; } = 512; //The threshold of maximum count of tokens per doc
        public int NumberOfSummaryTermsPerTopic { get; set; } = 10; //The number of words to summarize the topic
        public int NumberOfBurninIterations     { get; set; } = 10;
    }

    public class Lda : StorableObject<Lda, LDAModel>
    {
        public Lda(Language language, int version, string tag) : base(language, version, tag)
        {
        }

        public new static async Task<Lda> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new Lda(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Train(IEnumerable<IDocument> documents, int threads)
        {

        }



        /// <summary>
        /// Provide details about the topics discovered by <a href="https://arxiv.org/abs/1412.1576">LightLDA.</a>
        /// </summary>
        public sealed class ModelParameters
        {
            public struct ItemScore
            {
                public readonly int Item;
                public readonly float Score;
                public ItemScore(int item, float score)
                {
                    Item = item;
                    Score = score;
                }
            }
            public struct WordItemScore
            {
                public readonly int Item;
                public readonly string Word;
                public readonly float Score;
                public WordItemScore(int item, string word, float score)
                {
                    Item = item;
                    Word = word;
                    Score = score;
                }
            }

            // For each topic, provide information about the (item, score) pairs.
            public readonly IReadOnlyList<IReadOnlyList<ItemScore>> ItemScoresPerTopic;

            // For each topic, provide information about the (item, word, score) tuple.
            public readonly IReadOnlyList<IReadOnlyList<WordItemScore>> WordScoresPerTopic;

            internal ModelParameters(IReadOnlyList<IReadOnlyList<ItemScore>> itemScoresPerTopic)
            {
                ItemScoresPerTopic = itemScoresPerTopic;
            }

            internal ModelParameters(IReadOnlyList<IReadOnlyList<WordItemScore>> wordScoresPerTopic)
            {
                WordScoresPerTopic = wordScoresPerTopic;
            }
        }

        private sealed class LdaState : IDisposable
        {
            internal readonly LDAModel InfoEx;
            private readonly int _numVocab;
            private readonly object _preparationSyncRoot;
            private readonly object _testSyncRoot;
            private bool _predictionPreparationDone;
            private LdaSingleBox _ldaTrainer;

            private LdaState()
            {
                _preparationSyncRoot = new object();
                _testSyncRoot = new object();
            }

            internal LdaState(IExceptionContext ectx, LDAModel ex, int numVocab)
                : this()
            {
                InfoEx = ex;
                _numVocab = numVocab;

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumberOfTopics,
                    numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.MaximumNumberOfIterations,
                    InfoEx.LikelihoodInterval,
                    InfoEx.NumberOfThreads,
                    InfoEx.SamplingStepCount,
                    InfoEx.NumberOfSummaryTermsPerTopic,
                    false,
                    InfoEx.MaximumTokenCountPerDocument);
            }

            //internal LdaState(IExceptionContext ectx, ModelLoadContext ctx) : this()
            //{
            //    ectx.AssertValue(ctx);

            //    // *** Binary format ***
            //    // <ColInfoEx>
            //    // int: vocabnum
            //    // long: memblocksize
            //    // long: aliasMemBlockSize
            //    // (serializing term by term, for one term)
            //    // int: term_id, int: topic_num, KeyValuePair<int, int>[]: termTopicVector

            //    InfoEx = new LatentDirichletAllocationEstimator.ColumnOptions(ectx, ctx);

            //    _numVocab = ctx.Reader.ReadInt32();
            //    ectx.CheckDecode(_numVocab > 0);

            //    long memBlockSize = ctx.Reader.ReadInt64();
            //    ectx.CheckDecode(memBlockSize > 0);

            //    long aliasMemBlockSize = ctx.Reader.ReadInt64();
            //    ectx.CheckDecode(aliasMemBlockSize > 0);

            //    _ldaTrainer = new LdaSingleBox(
            //        InfoEx.NumberOfTopics,
            //        _numVocab, /* Need to set number of vocabulary here */
            //        InfoEx.AlphaSum,
            //        InfoEx.Beta,
            //        InfoEx.NumberOfIterations,
            //        InfoEx.LikelihoodInterval,
            //        InfoEx.NumberOfThreads,
            //        InfoEx.SamplingStepCount,
            //        InfoEx.NumberOfSummaryTermsPerTopic,
            //        false,
            //        InfoEx.MaximumTokenCountPerDocument);

            //    _ldaTrainer.AllocateModelMemory(_numVocab, InfoEx.NumberOfTopics, memBlockSize, aliasMemBlockSize);

            //    for (int i = 0; i < _numVocab; i++)
            //    {
            //        int termID = ctx.Reader.ReadInt32();
            //        ectx.CheckDecode(termID >= 0);
            //        int termTopicNum = ctx.Reader.ReadInt32();
            //        ectx.CheckDecode(termTopicNum >= 0);

            //        int[] topicId = new int[termTopicNum];
            //        int[] topicProb = new int[termTopicNum];

            //        for (int j = 0; j < termTopicNum; j++)
            //        {
            //            topicId[j] = ctx.Reader.ReadInt32();
            //            topicProb[j] = ctx.Reader.ReadInt32();
            //        }

            //        //set the topic into _ldaTrainer inner topic table
            //        _ldaTrainer.SetModel(termID, topicId, topicProb, termTopicNum);
            //    }

            //    //do the preparation
            //    if (!_predictionPreparationDone)
            //    {
            //        lock (_preparationSyncRoot)
            //        {
            //            _ldaTrainer.InitializeBeforeTest();
            //            _predictionPreparationDone = true;
            //        }
            //    }
            //}

            internal ModelParameters GetLdaSummary(VBuffer<ReadOnlyMemory<char>> mapping)
            {
                if (mapping.Length == 0)
                {
                    var itemScoresPerTopicBuilder = ImmutableArray.CreateBuilder<List<ModelParameters.ItemScore>>();
                    for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                    {
                        var scores = _ldaTrainer.GetTopicSummary(i);
                        var itemScores = new List<ModelParameters.ItemScore>();
                        foreach (KeyValuePair<int, float> p in scores)
                        {
                            itemScores.Add(new ModelParameters.ItemScore(p.Key, p.Value));
                        }

                        itemScoresPerTopicBuilder.Add(itemScores);
                    }
                    return new ModelParameters(itemScoresPerTopicBuilder.ToImmutable());
                }
                else
                {
                    ReadOnlyMemory<char> slotName = default;
                    var wordScoresPerTopicBuilder = ImmutableArray.CreateBuilder<List<ModelParameters.WordItemScore>>();
                    for (int i = 0; i < _ldaTrainer.NumTopic; i++)
                    {
                        var scores = _ldaTrainer.GetTopicSummary(i);
                        var wordScores = new List<ModelParameters.WordItemScore>();
                        foreach (KeyValuePair<int, float> p in scores)
                        {
                            mapping.GetItemOrDefault(p.Key, ref slotName);
                            wordScores.Add(new ModelParameters.WordItemScore(p.Key, slotName.ToString(), p.Value));
                        }
                        wordScoresPerTopicBuilder.Add(wordScores);
                    }
                    return new ModelParameters(wordScoresPerTopicBuilder.ToImmutable());
                }
            }

            //internal void Save(ModelSaveContext ctx)
            //{
            //    Debug.AssertValue(ctx);
            //    long memBlockSize = 0;
            //    long aliasMemBlockSize = 0;
            //    _ldaTrainer.GetModelStat(out memBlockSize, out aliasMemBlockSize);

            //    // *** Binary format ***
            //    // <ColInfoEx>
            //    // int: vocabnum
            //    // long: memblocksize
            //    // long: aliasMemBlockSize
            //    // (serializing term by term, for one term)
            //    // int: term_id, int: topic_num, KeyValuePair<int, int>[]: termTopicVector

            //    InfoEx.Save(ctx);
            //    ctx.Writer.Write(_ldaTrainer.NumVocab);
            //    ctx.Writer.Write(memBlockSize);
            //    ctx.Writer.Write(aliasMemBlockSize);

            //    //save model from this interface
            //    for (int i = 0; i < _ldaTrainer.NumVocab; i++)
            //    {
            //        KeyValuePair<int, int>[] termTopicVector = _ldaTrainer.GetModel(i);

            //        //write the topic to disk through ctx
            //        ctx.Writer.Write(i); //term_id
            //        ctx.Writer.Write(termTopicVector.Length);

            //        foreach (KeyValuePair<int, int> p in termTopicVector)
            //        {
            //            ctx.Writer.Write(p.Key);
            //            ctx.Writer.Write(p.Value);
            //        }
            //    }
            //}

            public void AllocateDataMemory(int docNum, long corpusSize)
            {
                _ldaTrainer.AllocateDataMemory(docNum, corpusSize);
            }

            public int FeedTrain(IExceptionContext ectx, in VBuffer<Double> input)
            {
                // REVIEW: Input the counts to your trainer here. This
                // is called multiple times.

                int docSize = 0;
                int termNum = 0;

                var inputValues = input.GetValues();
                for (int i = 0; i < inputValues.Length; i++)
                {
                    int termFreq = GetFrequency(inputValues[i]);
                    if (termFreq < 0)
                    {
                        // Ignore this row.
                        return 0;
                    }
                    if (docSize >= InfoEx.MaximumTokenCountPerDocument - termFreq)
                        break;

                    // If legal then add the term.
                    docSize += termFreq;
                    termNum++;
                }

                // Ignore empty doc.
                if (docSize == 0)
                    return 0;

                int actualSize = 0;
                if (input.IsDense)
                    actualSize = _ldaTrainer.LoadDocDense(inputValues, termNum, input.Length);
                else
                    actualSize = _ldaTrainer.LoadDoc(input.GetIndices(), inputValues, termNum, input.Length);

                return actualSize;
            }

            private static int GetFrequency(double value)
            {
                int result = (int)value;
                if (!(result == value && result >= 0))
                    return -1;
                return result;
            }
            public void CompleteTrain()
            {
                //allocate all kinds of in memory sample tables
                _ldaTrainer.InitializeBeforeTrain();

                //call native lda trainer to perform the multi-thread training
                _ldaTrainer.Train(""); /* Need to pass in an empty string */
            }

            public void Output(in VBuffer<Double> src, ref VBuffer<float> dst, int numBurninIter, bool reset)
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

                int len = InfoEx.NumberOfTopics;
                var srcValues = src.GetValues();
                if (srcValues.Length == 0)
                {
                    VBufferUtils.Resize(ref dst, len, 0);
                    return;
                }

                VBufferEditor<float> editor;
                // Make sure all the frequencies are valid and truncate if the sum gets too large.
                int docSize = 0;
                int termNum = 0;
                for (int i = 0; i < srcValues.Length; i++)
                {
                    int termFreq = GetFrequency(srcValues[i]);
                    if (termFreq < 0)
                    {
                        // REVIEW: Should this log a warning message? And what should it produce?
                        // It currently produces a vbuffer of all NA values.
                        // REVIEW: Need a utility method to do this...
                        editor = VBufferEditor.Create(ref dst, len);

                        for (int k = 0; k < len; k++)
                            editor.Values[k] = float.NaN;
                        dst = editor.Commit();
                        return;
                    }

                    if (docSize >= InfoEx.MaximumTokenCountPerDocument - termFreq)
                        break;

                    docSize += termFreq;
                    termNum++;
                }

                // REVIEW: Too much memory allocation here on each prediction.
                List<KeyValuePair<int, float>> retTopics;
                if (src.IsDense)
                    retTopics = _ldaTrainer.TestDocDense(srcValues, termNum, numBurninIter, reset);
                else
                    retTopics = _ldaTrainer.TestDoc(src.GetIndices(), srcValues, termNum, numBurninIter, reset);

                int count = retTopics.Count;
                Debug.Assert(count <= len);

                editor = VBufferEditor.Create(ref dst, len, count);
                double normalizer = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = retTopics[i].Key;
                    float value = retTopics[i].Value;
                    Debug.Assert(value >= 0);
                    Debug.Assert(0 <= index && index < len);
                    if (count < len)
                    {
                        Debug.Assert(i == 0 || editor.Indices[i - 1] < index);
                        editor.Indices[i] = index;
                    }
                    else
                        Debug.Assert(index == i);

                    editor.Values[i] = value;
                    normalizer += value;
                }

                if (normalizer > 0)
                {
                    for (int i = 0; i < count; i++)
                        editor.Values[i] = (float)(editor.Values[i] / normalizer);
                }

                dst = editor.Commit();
            }

            public void Dispose()
            {
                _ldaTrainer.Dispose();
            }
        }
    }
}
