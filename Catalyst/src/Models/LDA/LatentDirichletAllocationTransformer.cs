using Mosaik.Core;
using System;
using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML;
using System.Collections.Immutable;
using System.Linq;

namespace Catalyst.Models.LDA
{
    // LightLDA transform: Big Topic Models on Modest Compute Clusters.
    // <a href="https://arxiv.org/abs/1412.1576">LightLDA</a> is an implementation of Latent Dirichlet Allocation (LDA).
    // Previous implementations of LDA such as SparseLDA or AliasLDA allow to achieve massive data and model scales,
    // for example models with tens of billions of parameters to be inferred from billions of documents.
    // However this requires using a cluster of thousands of machines with all ensuing costs to setup and maintain.
    // LightLDA solves this problem in a more cost-effective manner by providing an implementation
    // that is efﬁcient enough for modest clusters with at most tens of machines...
    // For more details please see original LightLDA paper:
    // https://arxiv.org/abs/1412.1576
    // http://www.www2015.it/documents/proceedings/proceedings/p1351.pdf
    // and open source implementation:
    // https://github.com/Microsoft/LightLDA
    //
    // See <a href="https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.TestFramework/DataPipe/TestDataPipe.cs"/>
    // for an example on how to use LatentDirichletAllocationTransformer.
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="LatentDirichletAllocationEstimator"/>.
    /// </summary>
    public sealed class LatentDirichletAllocationTransformer
    {
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

        internal ModelParameters GetLdaDetails(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < _ldas.Length);

            var ldaState = _ldas[iinfo];
            var mapping = _columnMappings[iinfo];

            return ldaState.GetLdaSummary(mapping);
        }

        private sealed class LdaState : IDisposable
        {
            internal readonly LatentDirichletAllocationEstimator.ColumnOptions InfoEx;
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

            internal LdaState(IExceptionContext ectx, LatentDirichletAllocationEstimator.ColumnOptions ex, int numVocab)
                : this()
            {
                Contracts.AssertValue(ectx);
                ectx.AssertValue(ex, "ex");

                ectx.Assert(numVocab >= 0);
                InfoEx = ex;
                _numVocab = numVocab;

                _ldaTrainer = new LdaSingleBox(
                    InfoEx.NumberOfTopics,
                    numVocab, /* Need to set number of vocabulary here */
                    InfoEx.AlphaSum,
                    InfoEx.Beta,
                    InfoEx.NumberOfIterations,
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
//    Contracts.AssertValue(ctx);
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
                Contracts.AssertValue(ectx);

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

                ectx.Assert(actualSize == 2 * docSize + 1, string.Format("The doc size are distinct. Actual: {0}, Expected: {1}", actualSize, 2 * docSize + 1));
                return actualSize;
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
                Contracts.Assert(count <= len);

                editor = VBufferEditor.Create(ref dst, len, count);
                double normalizer = 0;
                for (int i = 0; i < count; i++)
                {
                    int index = retTopics[i].Key;
                    float value = retTopics[i].Value;
                    Contracts.Assert(value >= 0);
                    Contracts.Assert(0 <= index && index < len);
                    if (count < len)
                    {
                        Contracts.Assert(i == 0 || editor.Indices[i - 1] < index);
                        editor.Indices[i] = index;
                    }
                    else
                        Contracts.Assert(index == i);

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

        //private sealed class Mapper 
        //{
        //    private readonly LatentDirichletAllocationTransformer _parent;
        //    private readonly int[] _srcCols;

        //    public Mapper(LatentDirichletAllocationTransformer parent, DataViewSchema inputSchema) /*: base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)*/
        //    {
        //        _parent = parent;
        //        _srcCols = new int[_parent.ColumnPairs.Length];

        //        for (int i = 0; i < _parent.ColumnPairs.Length; i++)
        //        {
        //            if (!inputSchema.TryGetColumnIndex(_parent.ColumnPairs[i].inputColumnName, out _srcCols[i]))
        //                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName);

        //            var srcCol = inputSchema[_srcCols[i]];
        //            var srcType = srcCol.Type as VectorDataViewType;
        //            if (srcType == null || !srcType.IsKnownSize || !(srcType.ItemType is NumberDataViewType))
        //                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", _parent.ColumnPairs[i].inputColumnName, "known-size vector of Single", srcCol.Type.ToString());
        //        }
        //    }

        //    protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
        //    {
        //        var result = new DataViewSchema.DetachedColumn[_parent.ColumnPairs.Length];
        //        for (int i = 0; i < _parent.ColumnPairs.Length; i++)
        //        {
        //            var info = _parent._columns[i];
        //            result[i] = new DataViewSchema.DetachedColumn(_parent.ColumnPairs[i].outputColumnName, new VectorDataViewType(NumberDataViewType.Single, info.NumberOfTopics), null);
        //        }
        //        return result;
        //    }

        //    protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
        //    {
        //        Contracts.AssertValue(input);
        //        Contracts.Assert(0 <= iinfo && iinfo < _parent.ColumnPairs.Length);
        //        disposer = null;

        //        return GetTopic(input, iinfo);
        //    }

        //    private ValueGetter<VBuffer<float>> GetTopic(DataViewRow input, int iinfo)
        //    {
        //        var getSrc = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, input, _srcCols[iinfo]);
        //        var src = default(VBuffer<Double>);
        //        var lda = _parent._ldas[iinfo];
        //        int numBurninIter = lda.InfoEx.NumberOfBurninIterations;
        //        bool reset = lda.InfoEx.ResetRandomGenerator;
        //        return
        //            (ref VBuffer<float> dst) =>
        //            {
        //                // REVIEW: This will work, but there are opportunities for caching
        //                // based on input.Counter that are probably worthwhile given how long inference takes.
        //                getSrc(ref src);
        //                lda.Output(in src, ref dst, numBurninIter, reset);
        //            };
        //    }
        //}

        internal const string LoaderSignature = "LdaTransform";

        private readonly LatentDirichletAllocationEstimator.ColumnOptions[] _columns;
        private readonly LdaState[] _ldas;
        private readonly List<VBuffer<ReadOnlyMemory<char>>> _columnMappings;

        private const string RegistrationName = "LightLda";
        private const string WordTopicModelFilename = "word_topic_summary.txt";
        internal const string Summary = "The LDA transform implements LightLDA, a state-of-the-art implementation of Latent Dirichlet Allocation.";
        internal const string UserName = "Latent Dirichlet Allocation Transform";
        internal const string ShortName = "LightLda";

        private static (string outputColumnName, string inputColumnName)[] GetColumnPairs(LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            Contracts.CheckValue(columns, nameof(columns));
            return columns.Select(x => (x.Name, x.InputColumnName)).ToArray();
        }

        /// <summary>
        /// Initializes a new <see cref="LatentDirichletAllocationTransformer"/> object.
        /// </summary>
        /// <param name="env">Host Environment.</param>
        /// <param name="ldas">An array of LdaState objects, where ldas[i] is learnt from the i-th element of <paramref name="columns"/>.</param>
        /// <param name="columnMappings">A list of mappings, where columnMapping[i] is a map of slot names for the i-th element of <paramref name="columns"/>.</param>
        /// <param name="columns">Describes the parameters of the LDA process for each column pair.</param>
        private LatentDirichletAllocationTransformer(IHostEnvironment env, LdaState[] ldas, List<VBuffer<ReadOnlyMemory<char>>> columnMappings, params LatentDirichletAllocationEstimator.ColumnOptions[] columns) /*: base(Contracts.CheckRef(env, nameof(env)).Register(nameof(LatentDirichletAllocationTransformer)), GetColumnPairs(columns))*/
        {
            //Host.AssertNonEmpty(ColumnPairs);
            _ldas = ldas;
            _columnMappings = columnMappings;
            _columns = columns;
        }

//private LatentDirichletAllocationTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
//{
//    Host.AssertValue(ctx);

//    // *** Binary format ***
//    // <prefix handled in static Create method>
//    // <base>
//    // ldaState[num infos]: The LDA parameters

//    // Note: columnsLength would be just one in most cases.
//    var columnsLength = ColumnPairs.Length;
//    _columns = new LatentDirichletAllocationEstimator.ColumnOptions[columnsLength];
//    _ldas = new LdaState[columnsLength];
//    for (int i = 0; i < _ldas.Length; i++)
//    {
//        _ldas[i] = new LdaState(Host, ctx);
//        _columns[i] = _ldas[i].InfoEx;
//    }
//}

        internal static LatentDirichletAllocationTransformer TrainLdaTransformer(IHostEnvironment env, IDataView inputData, params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            var ldas = new LdaState[columns.Length];

            List<VBuffer<ReadOnlyMemory<char>>> columnMappings;
            using (var ch = env.Start("Train"))
            {
                columnMappings = Train(env, ch, inputData, ldas, columns);
            }

            return new LatentDirichletAllocationTransformer(env, ldas, columnMappings, columns);
        }

        private void Dispose(bool disposing)
        {
            if (_ldas != null)
            {
                foreach (var state in _ldas)
                    state?.Dispose();
            }
            if (disposing)
                GC.SuppressFinalize(this);
        }

        public void Dispose()
        {
            Dispose(true);
        }

        ~LatentDirichletAllocationTransformer()
        {
            Dispose(false);
        }

        //// Factory method for SignatureLoadDataTransform.
        //private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        //    => Create(env, ctx).MakeDataTransform(input);

        //// Factory method for SignatureLoadRowMapper.
        //private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
        //    => Create(env, ctx).MakeRowMapper(inputSchema);

        //// Factory method for SignatureDataTransform.
        //private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        //{
        //    Contracts.CheckValue(env, nameof(env));
        //    env.CheckValue(options, nameof(options));
        //    env.CheckValue(input, nameof(input));
        //    env.CheckValue(options.Columns, nameof(options.Columns));

        //    var cols = options.Columns.Select(colPair => new LatentDirichletAllocationEstimator.ColumnOptions(colPair, options)).ToArray();
        //    return TrainLdaTransformer(env, input, cols).MakeDataTransform(input);
        //}

        // Factory method for SignatureLoadModel
        //private static LatentDirichletAllocationTransformer Create(IHostEnvironment env, ModelLoadContext ctx)
        //{
        //    Contracts.CheckValue(env, nameof(env));
        //    var h = env.Register(RegistrationName);

        //    h.CheckValue(ctx, nameof(ctx));
        //    ctx.CheckAtModel(GetVersionInfo());

        //    return h.Apply(
        //        "Loading Model",
        //        ch =>
        //        {
        //            // *** Binary Format ***
        //            // int: sizeof(float)
        //            // <remainder handled in ctors>
        //            int cbFloat = ctx.Reader.ReadInt32();
        //            h.CheckDecode(cbFloat == sizeof(float));
        //            return new LatentDirichletAllocationTransformer(h, ctx);
        //        });
        //}

        //private protected override void SaveModel(ModelSaveContext ctx)
        //{
        //    Host.CheckValue(ctx, nameof(ctx));
        //    ctx.CheckAtModel();
        //    ctx.SetVersionInfo(GetVersionInfo());

        //    // *** Binary format ***
        //    // int: sizeof(float)
        //    // <base>
        //    // ldaState[num infos]: The LDA parameters

        //    ctx.Writer.Write(sizeof(float));
        //    SaveColumns(ctx);
        //    for (int i = 0; i < _ldas.Length; i++)
        //    {
        //        _ldas[i].Save(ctx);
        //    }
        //}

        private static int GetFrequency(double value)
        {
            int result = (int)value;
            if (!(result == value && result >= 0))
                return -1;
            return result;
        }

        private static List<VBuffer<ReadOnlyMemory<char>>> Train(IHostEnvironment env, IChannel ch, IDataView inputData, LdaState[] states, params LatentDirichletAllocationEstimator.ColumnOptions[] columns)
        {
            env.AssertValue(ch);
            ch.AssertValue(inputData);
            ch.AssertValue(states);
            ch.Assert(states.Length == columns.Length);

            var activeColumns = new List<DataViewSchema.Column>();
            int[] numVocabs = new int[columns.Length];
            int[] srcCols = new int[columns.Length];

            var columnMappings = new List<VBuffer<ReadOnlyMemory<char>>>();

            var inputSchema = inputData.Schema;
            for (int i = 0; i < columns.Length; i++)
            {
                if (!inputData.Schema.TryGetColumnIndex(columns[i].InputColumnName, out int srcCol))
                    throw env.ExceptSchemaMismatch(nameof(inputData), "input", columns[i].InputColumnName);

                var srcColType = inputSchema[srcCol].Type as VectorDataViewType;
                if (srcColType == null || !srcColType.IsKnownSize || !(srcColType.ItemType is NumberDataViewType))
                    throw env.ExceptSchemaMismatch(nameof(inputSchema), "input", columns[i].InputColumnName, "known-size vector of Single", srcColType.ToString());

                srcCols[i] = srcCol;
                activeColumns.Add(inputData.Schema[srcCol]);
                numVocabs[i] = 0;

                VBuffer<ReadOnlyMemory<char>> dst = default;
                if (inputSchema[srcCol].HasSlotNames(srcColType.Size))
                    inputSchema[srcCol].Annotations.GetValue(AnnotationUtils.Kinds.SlotNames, ref dst);
                else
                    dst = default(VBuffer<ReadOnlyMemory<char>>);
                columnMappings.Add(dst);
            }

            //the current lda needs the memory allocation before feedin data, so needs two sweeping of the data,
            //one for the pre-calc memory, one for feedin data really
            //another solution can be prepare these two value externally and put them in the beginning of the input file.
            long[] corpusSize = new long[columns.Length];
            int[] numDocArray = new int[columns.Length];

            using (var cursor = inputData.GetRowCursor(activeColumns))
            {
                var getters = new ValueGetter<VBuffer<Double>>[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    corpusSize[i] = 0;
                    numDocArray[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, cursor, srcCols[i]);
                }
                VBuffer<Double> src = default;
                long rowCount = 0;
                while (cursor.MoveNext())
                {
                    ++rowCount;
                    for (int i = 0; i < columns.Length; i++)
                    {
                        int docSize = 0;
                        getters[i](ref src);

                        // compute term, doc instance#.
                        var srcValues = src.GetValues();
                        for (int termID = 0; termID < srcValues.Length; termID++)
                        {
                            int termFreq = GetFrequency(srcValues[termID]);
                            if (termFreq < 0)
                            {
                                // Ignore this row.
                                docSize = 0;
                                break;
                            }

                            if (docSize >= columns[i].MaximumTokenCountPerDocument - termFreq)
                                break; //control the document length

                            //if legal then add the term
                            docSize += termFreq;
                        }

                        // Ignore empty doc
                        if (docSize == 0)
                            continue;

                        numDocArray[i]++;
                        corpusSize[i] += docSize * 2 + 1;   // in the beggining of each doc, there is a cursor variable

                        // increase numVocab if needed.
                        if (numVocabs[i] < src.Length)
                            numVocabs[i] = src.Length;
                    }
                }

                // No data to train on, just return
                if (rowCount == 0)
                    return columnMappings;

                for (int i = 0; i < columns.Length; ++i)
                {
                    if (numDocArray[i] != rowCount)
                    {
                        ch.Assert(numDocArray[i] < rowCount);
                        //ch.Warning($"Column '{columns[i].InputColumnName}' has skipped {rowCount - numDocArray[i]} of {rowCount} rows either empty or with negative, non-finite, or fractional values.");
                    }
                }
            }

            // Initialize all LDA states
            for (int i = 0; i < columns.Length; i++)
            {
                var state = new LdaState(env, columns[i], numVocabs[i]);

                if (numDocArray[i] == 0 || corpusSize[i] == 0)
                    throw ch.Except("The specified documents are all empty in column '{0}'.", columns[i].InputColumnName);

                state.AllocateDataMemory(numDocArray[i], corpusSize[i]);
                states[i] = state;
            }

            using (var cursor = inputData.GetRowCursor(activeColumns))
            {
                int[] docSizeCheck = new int[columns.Length];
                // This could be optimized so that if multiple trainers consume the same column, it is
                // fed into the train method once.
                var getters = new ValueGetter<VBuffer<Double>>[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    docSizeCheck[i] = 0;
                    getters[i] = RowCursorUtils.GetVecGetterAs<Double>(NumberDataViewType.Double, cursor, srcCols[i]);
                }

                VBuffer<double> src = default;

                while (cursor.MoveNext())
                {
                    for (int i = 0; i < columns.Length; i++)
                    {
                        getters[i](ref src);
                        docSizeCheck[i] += states[i].FeedTrain(env, in src);
                    }
                }

                for (int i = 0; i < columns.Length; i++)
                {
                    env.Assert(corpusSize[i] == docSizeCheck[i]);
                    states[i].CompleteTrain();
                }
            }

            return columnMappings;
        }

        //private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);
    }
}
