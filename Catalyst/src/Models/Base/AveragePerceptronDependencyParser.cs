using UID;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace Catalyst.Models
{
    public class AveragePerceptronDependencyParserModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public ConcurrentDictionary<int, float[]> Weights { get; set; }

        public Dictionary<int, string> Actions { get; set; }
        public Dictionary<string, int> Action2Index { get; set; }
        public Dictionary<int, int> Index2Move { get; set; }
        public Dictionary<int, string> Index2Label { get; set; }
    }

    public class AveragePerceptronDependencyParser : StorableObject<AveragePerceptronDependencyParser, AveragePerceptronDependencyParserModel>, IProcess
    {
        private const int N_ACTIONS = 3;

        private const int SHIFT = 0;
        private const int RIGHT = 1;
        private const int LEFT = 2;

        private ConcurrentDictionary<int, float[]> AverageWeights { get; set; }

        public AveragePerceptronDependencyParser(Language language, int version, string tag = "") : base(language, version, tag, compress: true)
        {
            Data.Weights = new ConcurrentDictionary<int, float[]>();
        }

        public new static async Task<AveragePerceptronDependencyParser> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new AveragePerceptronDependencyParser(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Train(IEnumerable<IDocument> documents, int trainingSteps = 10, float learningRate = 0.9f)
        {
            AverageWeights = new ConcurrentDictionary<int, float[]>();

            var sentences = documents.SelectMany(d => d.Spans).Where(s => s.IsProjective() && s.TokensCount > 4 && !s.Any(tk => tk.Value.Contains("@") || tk.Value.Contains("://"))).ToList();

            var sw = new System.Diagnostics.Stopwatch();

            var spanGolds = sentences.Select(span => (indexes: span.Select(tk => tk.Head + 1).ToArray(), labels: span.Select(tk => tk.DependencyType).ToArray())).ToList();
            var spanBuffers = sentences.Select(span => new Buffer(span, N_FEATURES, N_ACTIONS)).ToList();

            //Pad gold with ROOT and EOS tokens
            for (int i = 0; i < spanGolds.Count; i++)
            {
                var g = spanGolds[i];
                var tmpGold = new List<int>(g.indexes.Length + 2);
                var tmpLbl = new List<string>(tmpGold.Count);
                tmpGold.Add(-2); tmpGold.AddRange(g.indexes); tmpGold.Add(-2);
                tmpLbl.Add("<ROOT>"); tmpLbl.AddRange(g.labels); tmpLbl.Add("<EOS>");
                g.indexes = tmpGold.ToArray();
                g.labels = tmpLbl.ToArray();
                spanGolds[i] = g;
            }

            int nThreads = Environment.ProcessorCount;

            Console.WriteLine($"Runnning training with {nThreads} threads");

            int groupSize = (int)(spanBuffers.Count / nThreads);
            float factor = 1f;
            for (int step = 0; step < trainingSteps; step++)
            {
                spanBuffers.ShuffleTogether(spanGolds);

                int correct = 0, total = 0;

                //-------------------------------------------

                var threads = new Thread[nThreads];
                var bufferGroups = new List<List<Buffer>>();
                var goldsGroups = new List<List<(int[] indexes, string[] labels)>>();

                for (int i = 0; i < nThreads; i++)
                {
                    bufferGroups.Add(spanBuffers.Skip(groupSize * i).Take(groupSize).ToList());
                    goldsGroups.Add(spanGolds.Skip(groupSize * i).Take(groupSize).ToList());
                }

                for (int i = 0; i < nThreads; i++)
                {
                    threads[i] = new Thread((N) =>
                    {
                        int corr = 0, tot = 0, n = (int)N;
                        for (int j = 0; j < bufferGroups[n].Count; j++)
                        {
                            var sb = bufferGroups[n][j];
                            var golds = goldsGroups[n][j];
                            corr += TrainOnSentence(ref sb, golds.indexes, golds.labels, step, learningRate);
                            tot += goldsGroups[n][j].indexes.Length;
                        }
                        Interlocked.Add(ref correct, corr);
                        Interlocked.Add(ref total, tot);
                    });
                }

                sw.Restart();
                for (int i = 0; i < nThreads; i++) { threads[i].Priority = ThreadPriority.Highest; threads[i].Start(i); }
                for (int i = 0; i < nThreads; i++) { threads[i].Join(); }
                sw.Stop();

                Console.WriteLine($"{Languages.EnumToCode(Language)} Step {step + 1}/{trainingSteps} f=[{factor:0.00}]: {Math.Round(100D * correct / total, 2)}% @ {Math.Round(1000D * total / sw.ElapsedMilliseconds, 0) } tokens/second");
                UpdateAverages(epoch: step, final: (step == trainingSteps - 1));
                factor *= learningRate;
            }

            Data.Weights = AverageWeights;
            Data.TrainedTime = DateTime.UtcNow;
            AverageWeights = null;
        }

        internal int TrainOnSentence(ref Buffer buffer, int[] goldHeads, string[] goldLabels, int epoch, float learningRate)
        {
            buffer.CurrentIndex = 1; buffer.Stack.Clear(); buffer.Stack.Push(0);
            if (buffer.N < 4) { return buffer.N - 2; } // Nothing to do

            //if(false)
            //{
            //    var beamSearch = new BeamSearch<Buffer>(buffer, 4, (b) => TrainerGeneratorFromBuffer(b), (b, ia) => TrainerAdvanceBuffer(b, ia, goldHeads));

            //    while (!beamSearch.Finished)
            //    {
            //        beamSearch.Search();
            //        beamSearch.AdvanceBeam();
            //    }

            //    var result = beamSearch.GetResult();

            //    buffer.Heads = result.Heads;
            //    buffer.Labels = result.Labels;
            //}
            //else
            //{
            while (buffer.Stack.Length > 0 || buffer.CurrentIndex < buffer.N - 1)
            {
                GetFeatures(buffer.CurrentIndex, buffer.Hashes.Length, ref buffer);

                (bool CanLeft, bool CanRight, bool CanShift) = GetValidMoves(buffer.CurrentIndex, buffer.N, buffer.Stack.Length);
                (bool LeftIsGold, bool RightIsGold, bool ShiftIsGold) = GetGoldMoves(buffer.CurrentIndex, buffer.N, ref buffer, goldHeads);

                ComputeWeights(ref buffer);

                int predictedMove = GetBestMove(CanLeft, CanRight, CanShift, ref buffer);
                int goldMove = GetBestMove(LeftIsGold, RightIsGold, ShiftIsGold, ref buffer);

                UpdateModel(goldMove, predictedMove, ref buffer, learningRate);

                buffer.CurrentIndex = DoMove(predictedMove, buffer.CurrentIndex, ref buffer);
            }
            //}

            int correct = 0;
            for (int j = 1; j < buffer.N - 1; j++) { if (buffer.Heads[j] == buffer.N) { buffer.Heads[j] = 0; } if (buffer.Heads[j] == goldHeads[j]) { correct++; } }
            return correct;
        }

        internal void Predict(ref Buffer buffer)
        {
            buffer.CurrentIndex = 1; buffer.Stack.Clear(); buffer.Stack.Push(0);
            if (buffer.N < 4) { return; } // Nothing to do

            //var beamSearch = new BeamSearch<Buffer>(buffer, 4, (b) => GeneratorFromBuffer(b), (b, ia) => AdvanceBuffer(b, ia));

            //while (!beamSearch.Finished)
            //{
            //    beamSearch.Search();
            //    beamSearch.AdvanceBeam();
            //}

            //var result = beamSearch.GetResult();

            //buffer.Heads = result.Heads;
            //buffer.Labels = result.Labels;

            while (buffer.Stack.Length > 0 || buffer.CurrentIndex < buffer.N - 1)
            {
                GetFeatures(buffer.CurrentIndex, buffer.N, ref buffer);
                (bool CanLeft, bool CanRight, bool CanShift) = GetValidMoves(buffer.CurrentIndex, buffer.N, buffer.Stack.Length);
                ComputeWeights(ref buffer);

                int predictedMove = GetBestMove(CanLeft, CanRight, CanShift, ref buffer);

                buffer.CurrentIndex = DoMove(predictedMove, buffer.CurrentIndex, ref buffer);
            }
        }

        private void AdvanceBuffer(Buffer buffer, int move)
        {
            buffer.CurrentIndex = DoMove(move, buffer.CurrentIndex, ref buffer);
        }

        private void TrainerAdvanceBuffer(Buffer buffer, int move, int[] goldHeads)
        {
            //(bool LeftIsGold, bool RightIsGold, bool ShiftIsGold) = GetGoldMoves(buffer.CurrentIndex, buffer.Hashes.Length - 1, ref buffer, goldHeads);
            //int goldMove = GetBestMove(LeftIsGold, RightIsGold, ShiftIsGold, ref buffer);
            //UpdateModel(goldMove, move, ref buffer);
            //buffer.CurrentIndex = DoMove(move, buffer.CurrentIndex, ref buffer);
        }

        private IList<BeamAction> TrainerGeneratorFromBuffer(Buffer buffer)
        {
            if (!(buffer.Stack.Length > 0 || buffer.CurrentIndex < buffer.Hashes.Length - 1)) { return null; }
            var actions = new List<BeamAction>();

            //GetFeatures(buffer.CurrentIndex, buffer.Hashes.Length, ref buffer);
            //(bool CanLeft, bool CanRight, bool CanShift) = GetValidMoves(buffer.CurrentIndex, buffer.Hashes.Length, buffer.Stack.Length);

            //ComputeWeights(ref buffer);

            //var (moves, scores) = GetBestMoves(CanLeft, CanRight, CanShift, ref buffer);
            //for (int i = 0; i < moves.Length; i++)
            //{
            //    actions.Add(new BeamAction(moves[i], scores[i]));
            //}

            return actions;
        }

        private IList<BeamAction> GeneratorFromBuffer(Buffer buffer)
        {
            if (!(buffer.Stack.Length > 0 || buffer.CurrentIndex < buffer.N - 1)) { return null; }
            var actions = new List<BeamAction>();

            GetFeatures(buffer.CurrentIndex, buffer.N, ref buffer);
            (bool CanLeft, bool CanRight, bool CanShift) = GetValidMoves(buffer.CurrentIndex, buffer.N, buffer.Stack.Length);
            ComputeWeights(ref buffer);

            var (moves, scores) = GetBestMoves(CanLeft, CanRight, CanShift, ref buffer);
            for (int i = 0; i < moves.Length; i++)
            {
                actions.Add(new BeamAction(moves[i], scores[i]));
            }

            return actions;
        }

        #region FEATURES

        private const int N_FEATURES = 66 + 6 + 6 + 1 + 6 + 6 - 18;
        private static readonly int _HashEmpty = GetHash("<EMPTY>");
        private static readonly int _HashBias = GetHash("<BIAS>");
        private static readonly int _HashPosition = GetHash("<POSITION>");
        private static readonly int _HashWs0 = GetHash("Ws0");
        private static readonly int _HashWs1 = GetHash("Ws1");
        private static readonly int _HashWs2 = GetHash("Ws2");
        private static readonly int _HashTs0 = GetHash("Ts0");
        private static readonly int _HashTs1 = GetHash("Ts1");
        private static readonly int _HashTs2 = GetHash("Ts2");
        private static readonly int _HashWn0 = GetHash("Wn0");
        private static readonly int _HashWn1 = GetHash("Wn1");
        private static readonly int _HashWn2 = GetHash("Wn2");
        private static readonly int _HashTn0 = GetHash("Tn0");
        private static readonly int _HashTn1 = GetHash("Tn1");
        private static readonly int _HashTn2 = GetHash("Tn2");
        private static readonly int _HashVn0b = GetHash("Vn0b");
        private static readonly int _HashWn0b1 = GetHash("Wn0b1");
        private static readonly int _HashWn0b2 = GetHash("Wn0b2");
        private static readonly int _HashTn0b1 = GetHash("Tn0b1");
        private static readonly int _HashTn0b2 = GetHash("Tn0b2");
        private static readonly int _HashVn0f = GetHash("Vn0f");
        private static readonly int _HashWn0f1 = GetHash("Wn0f1");
        private static readonly int _HashWn0f2 = GetHash("Wn0f2");
        private static readonly int _HashTn0f1 = GetHash("Tn0f1");
        private static readonly int _HashTn0f2 = GetHash("Tn0f2");
        private static readonly int _HashVs0b = GetHash("Vs0b");
        private static readonly int _HashWs0b1 = GetHash("Ws0b1");
        private static readonly int _HashWs0b2 = GetHash("Ws0b2");
        private static readonly int _HashTs0b1 = GetHash("Ts0b1");
        private static readonly int _HashTs0b2 = GetHash("Ts0b2");
        private static readonly int _HashVs0f = GetHash("Vs0f");
        private static readonly int _HashWs0f1 = GetHash("Ws0f1");
        private static readonly int _HashWs0f2 = GetHash("Ws0f2");
        private static readonly int _HashTs0f1 = GetHash("Ts0f1");
        private static readonly int _HashTs0f2 = GetHash("Ts0f2");
        private static readonly int _HashWn0_Tn0 = GetHash("Wn0_Tn0");
        private static readonly int _HashWn1_Tn1 = GetHash("Wn1_Tn1");
        private static readonly int _HashWn2_Tn2 = GetHash("Wn2_Tn2");
        private static readonly int _HashWs0_Ts0 = GetHash("Ws0_Ts0");
        private static readonly int _HashWs0_Wn0 = GetHash("Ws0_Wn0");
        private static readonly int _HashWn0_Tn0_Ws0 = GetHash("Wn0_Tn0_Ws0");
        private static readonly int _HashWn0_Tn0_Ts0 = GetHash("Wn0_Tn0_Ts0");
        private static readonly int _HashWs0_Ts0_Wn0 = GetHash("Ws0_Ts0_Wn0");
        private static readonly int _HashWs0_Ts0_Tn0 = GetHash("Ws0_Ts0_Tn0");
        private static readonly int _HashWs0_Ts0_Wn0_Tn0 = GetHash("Ws0_Ts0_Wn0_Tn0");
        private static readonly int _HashTs0_Tn0 = GetHash("Ts0_Tn0");
        private static readonly int _HashTn0_Tn1 = GetHash("Tn0_Tn1");
        private static readonly int _HashTn0_Tn1_Tn2 = GetHash("Tn0_Tn1_Tn2");
        private static readonly int _HashTs0_Tn0_Tn1 = GetHash("Ts0_Tn0_Tn1");
        private static readonly int _HashTs0_Ts1_Tn0 = GetHash("Ts0_Ts1_Tn0");
        private static readonly int _HashTs0_Ts0f1_Tn0 = GetHash("Ts0_Ts0f1_Tn0");
        private static readonly int _HashTs0_Tn0_Tn0b1 = GetHash("Ts0_Tn0_Tn0b1");
        private static readonly int _HashTs0_Ts0b1_Ts0b2 = GetHash("Ts0_Ts0b1_Ts0b2");
        private static readonly int _HashTs0_Ts0f1_Ts0f2 = GetHash("Ts0_Ts0f1_Ts0f2");
        private static readonly int _HashTn0_Tn0b1_Tn0b2 = GetHash("Tn0_Tn0b1_Tn0b2");
        private static readonly int _HashTs0_Ts1_Ts1 = GetHash("Ts0_Ts1_Ts1");
        private static readonly int _HashWs0_Vs0f = GetHash("Ws0_Vs0f");
        private static readonly int _HashWs0_Vs0b = GetHash("Ws0_Vs0b");
        private static readonly int _HashWn0_Vn0b = GetHash("Wn0_Vn0b");
        private static readonly int _HashTs0_Vs0f = GetHash("Ts0_Vs0f");
        private static readonly int _HashTs0_Vs0b = GetHash("Ts0_Vs0b");
        private static readonly int _HashTn0_Vn0b = GetHash("Tn0_Vn0b");
        private static readonly int _HashWs0_Ds0n0 = GetHash("Ws0_Ds0n0");
        private static readonly int _HashWn0_Ds0n0 = GetHash("Wn0_Ds0n0");
        private static readonly int _HashTs0_Ds0n0 = GetHash("Ts0_Ds0n0");
        private static readonly int _HashTn0_Ds0n0 = GetHash("Tn0_Ds0n0");
        private static readonly int _Hash_t_Tn0_Ts0_Ds0n0 = GetHash("'t' + Tn0+Ts0_Ds0n0");
        private static readonly int _Hash_w_Wn0_Ws0_Ds0n0 = GetHash("'w' + Wn0+Ws0_Ds0n0");

        private static readonly int _HashWs0_Suffix = GetHash("Ws0_Suffix");
        private static readonly int _HashWs1_Suffix = GetHash("Ws1_Suffix");
        private static readonly int _HashWs2_Suffix = GetHash("Ws2_Suffix");
        private static readonly int _HashWn0_Suffix = GetHash("Wn0_Suffix");
        private static readonly int _HashWn1_Suffix = GetHash("Wn1_Suffix");
        private static readonly int _HashWn2_Suffix = GetHash("Wn2_Suffix");

        private static readonly int _HashWs0_Prefix = GetHash("Ws0_Prefix");
        private static readonly int _HashWs1_Prefix = GetHash("Ws1_Prefix");
        private static readonly int _HashWs2_Prefix = GetHash("Ws2_Prefix");
        private static readonly int _HashWn0_Prefix = GetHash("Wn0_Prefix");
        private static readonly int _HashWn1_Prefix = GetHash("Wn1_Prefix");
        private static readonly int _HashWn2_Prefix = GetHash("Wn2_Prefix");

        private static readonly int _HashWs0_HasNumeric = GetHash("Ws0_HasNumeric");
        private static readonly int _HashWs1_HasNumeric = GetHash("Ws1_HasNumeric");
        private static readonly int _HashWs2_HasNumeric = GetHash("Ws2_HasNumeric");
        private static readonly int _HashWn0_HasNumeric = GetHash("Wn0_HasNumeric");
        private static readonly int _HashWn1_HasNumeric = GetHash("Wn1_HasNumeric");
        private static readonly int _HashWn2_HasNumeric = GetHash("Wn2_HasNumeric");

        private static readonly int _HashWs0_Shape = GetHash("Ws0_Shape");
        private static readonly int _HashWs1_Shape = GetHash("Ws1_Shape");
        private static readonly int _HashWs2_Shape = GetHash("Ws2_Shape");
        private static readonly int _HashWn0_Shape = GetHash("Wn0_Shape");
        private static readonly int _HashWn1_Shape = GetHash("Wn1_Shape");
        private static readonly int _HashWn2_Shape = GetHash("Wn2_Shape");

        #endregion FEATURES

        private void GetFeatures(int i, int n, ref Buffer buffer)
        {
            int depth = buffer.Stack.Length;
            int s0 = depth > 0 ? buffer.Stack[0] : -1;

            var (Ws0, Ws1, Ws2) = GetStackContext(depth, ref buffer.Stack, ref buffer.Hashes);
            //var (Ws0_s, Ws1_s, Ws2_s) = GetStackContext(depth, ref buffer.Stack, ref buffer.SuffixHashes);
            //var (Ws0_p, Ws1_p, Ws2_p) = GetStackContext(depth, ref buffer.Stack, ref buffer.PrefixHashes);
            //var (Ws0_n, Ws1_n, Ws2_n) = GetStackContext(depth, ref buffer.Stack, ref buffer.NumericHashes);
            var (Ws0_x, Ws1_x, Ws2_x) = GetStackContext(depth, ref buffer.Stack, ref buffer.ShapeHashes);
            var (Ts0, Ts1, Ts2) = GetStackContext(depth, ref buffer.Stack, ref buffer.TagHashes);

            var (Wn0, Wn1, Wn2) = GetBufferContext(i, n, ref buffer.Hashes);
            //var (Wn0_s, Wn1_s, Wn2_s) = GetBufferContext(i, n, ref buffer.SuffixHashes);
            //var (Wn0_p, Wn1_p, Wn2_p) = GetBufferContext(i, n, ref buffer.PrefixHashes);
            //var (Wn0_n, Wn1_n, Wn2_n) = GetBufferContext(i, n, ref buffer.NumericHashes);
            var (Wn0_x, Wn1_x, Wn2_x) = GetBufferContext(i, n, ref buffer.ShapeHashes);
            var (Tn0, Tn1, Tn2) = GetBufferContext(i, n, ref buffer.TagHashes);

            var (Vn0b, Wn0b1, Wn0b2) = GetParseContext(i, ref buffer.Lefts, ref buffer.Hashes);
            var (_, Tn0b1, Tn0b2) = GetParseContext(i, ref buffer.Lefts, ref buffer.TagHashes);

            var (Vn0f, Wn0f1, Wn0f2) = GetParseContext(i, ref buffer.Rights, ref buffer.Hashes);
            var (_, Tn0f1, Tn0f2) = GetParseContext(i, ref buffer.Rights, ref buffer.TagHashes);

            var (Vs0b, Ws0b1, Ws0b2) = GetParseContext(i, ref buffer.Lefts, ref buffer.Hashes);
            var (_, Ts0b1, Ts0b2) = GetParseContext(i, ref buffer.Lefts, ref buffer.TagHashes);

            var (Vs0f, Ws0f1, Ws0f2) = GetParseContext(i, ref buffer.Rights, ref buffer.Hashes);
            var (_, Ts0f1, Ts0f2) = GetParseContext(i, ref buffer.Rights, ref buffer.TagHashes);

            //String-distance
            int Ds0n0 = s0 > 0 ? Math.Min(i - s0, 5) : 0;

            int k = 0;

            for (int h = 0; h < N_FEATURES; h++) { buffer.Features[h] = _HashEmpty; } //Clears all features

            buffer.Features[k++] = _HashBias;
            buffer.Features[k++] = HashCombine(_HashPosition, i);
            buffer.Features[k++] = HashCombine(_HashWs0, Ws0);
            buffer.Features[k++] = HashCombine(_HashWs1, Ws1);
            buffer.Features[k++] = HashCombine(_HashWs2, Ws2);
            buffer.Features[k++] = HashCombine(_HashTs0, Ts0);
            buffer.Features[k++] = HashCombine(_HashTs1, Ts1);
            buffer.Features[k++] = HashCombine(_HashTs2, Ts2);
            buffer.Features[k++] = HashCombine(_HashWn0, Wn0);
            buffer.Features[k++] = HashCombine(_HashWn1, Wn1);
            buffer.Features[k++] = HashCombine(_HashWn2, Wn2);
            buffer.Features[k++] = HashCombine(_HashTn0, Tn0);
            buffer.Features[k++] = HashCombine(_HashTn1, Tn1);
            buffer.Features[k++] = HashCombine(_HashTn2, Tn2);

            //buffer.Features[k++] = HashCombine(_HashWs0_Suffix,   Ws0_s);
            //buffer.Features[k++] = HashCombine(_HashWs1_Suffix,   Ws1_s);
            //buffer.Features[k++] = HashCombine(_HashWs2_Suffix,   Ws2_s);
            //buffer.Features[k++] = HashCombine(_HashWn0_Suffix,   Wn0_s);
            //buffer.Features[k++] = HashCombine(_HashWn1_Suffix,   Wn1_s);
            //buffer.Features[k++] = HashCombine(_HashWn2_Suffix,   Wn2_s);

            //buffer.Features[k++] = HashCombine(_HashWs0_Prefix,   Ws0_p);
            //buffer.Features[k++] = HashCombine(_HashWs1_Prefix,   Ws1_p);
            //buffer.Features[k++] = HashCombine(_HashWs2_Prefix,   Ws2_p);
            //buffer.Features[k++] = HashCombine(_HashWn0_Prefix,   Wn0_p);
            //buffer.Features[k++] = HashCombine(_HashWn1_Prefix,   Wn1_p);
            //buffer.Features[k++] = HashCombine(_HashWn2_Prefix,   Wn2_p);

            //buffer.Features[k++] = HashCombine(_HashWs0_HasNumeric,   Ws0_n);
            //buffer.Features[k++] = HashCombine(_HashWs1_HasNumeric,   Ws1_n);
            //buffer.Features[k++] = HashCombine(_HashWs2_HasNumeric,   Ws2_n);
            //buffer.Features[k++] = HashCombine(_HashWn0_HasNumeric,   Wn0_n);
            //buffer.Features[k++] = HashCombine(_HashWn1_HasNumeric,   Wn1_n);
            //buffer.Features[k++] = HashCombine(_HashWn2_HasNumeric,   Wn2_n);

            buffer.Features[k++] = HashCombine(_HashWs0_Shape, Ws0_x);
            buffer.Features[k++] = HashCombine(_HashWs1_Shape, Ws1_x);
            buffer.Features[k++] = HashCombine(_HashWs2_Shape, Ws2_x);
            buffer.Features[k++] = HashCombine(_HashWn0_Shape, Wn0_x);
            buffer.Features[k++] = HashCombine(_HashWn1_Shape, Wn1_x);
            buffer.Features[k++] = HashCombine(_HashWn2_Shape, Wn2_x);

            buffer.Features[k++] = HashCombine(_HashVn0b, Vn0b);
            buffer.Features[k++] = HashCombine(_HashWn0b1, Wn0b1);
            buffer.Features[k++] = HashCombine(_HashWn0b2, Wn0b2);
            buffer.Features[k++] = HashCombine(_HashTn0b1, Tn0b1);
            buffer.Features[k++] = HashCombine(_HashTn0b2, Tn0b2);
            buffer.Features[k++] = HashCombine(_HashVn0f, Vn0f);
            buffer.Features[k++] = HashCombine(_HashWn0f1, Wn0f1);
            buffer.Features[k++] = HashCombine(_HashWn0f2, Wn0f2);
            buffer.Features[k++] = HashCombine(_HashTn0f1, Tn0f1);
            buffer.Features[k++] = HashCombine(_HashTn0f2, Tn0f2);
            buffer.Features[k++] = HashCombine(_HashVs0b, Vs0b);
            buffer.Features[k++] = HashCombine(_HashWs0b1, Ws0b1);
            buffer.Features[k++] = HashCombine(_HashWs0b2, Ws0b2);
            buffer.Features[k++] = HashCombine(_HashTs0b1, Ts0b1);
            buffer.Features[k++] = HashCombine(_HashTs0b2, Ts0b2);
            buffer.Features[k++] = HashCombine(_HashVs0f, Vs0f);
            buffer.Features[k++] = HashCombine(_HashWs0f1, Ws0f1);
            buffer.Features[k++] = HashCombine(_HashWs0f2, Ws0f2);
            buffer.Features[k++] = HashCombine(_HashTs0f1, Ts0f1);
            buffer.Features[k++] = HashCombine(_HashTs0f2, Ts0f2);

            buffer.Features[k++] = HashCombine(_HashWn0_Tn0, HashCombine(Wn0, Tn0));
            buffer.Features[k++] = HashCombine(_HashWn1_Tn1, HashCombine(Wn1, Tn1));
            buffer.Features[k++] = HashCombine(_HashWn2_Tn2, HashCombine(Wn2, Tn2));
            buffer.Features[k++] = HashCombine(_HashWs0_Ts0, HashCombine(Ws0, Ts0));

            buffer.Features[k++] = HashCombine(_HashWs0_Wn0, HashCombine(Ws0, Wn0));
            buffer.Features[k++] = HashCombine(_HashWn0_Tn0_Ws0, HashCombine(Wn0, Tn0, Ws0));
            buffer.Features[k++] = HashCombine(_HashWn0_Tn0_Ts0, HashCombine(Wn0, Tn0, Ts0));
            buffer.Features[k++] = HashCombine(_HashWs0_Ts0_Wn0, HashCombine(Ws0, Ts0, Wn0));
            buffer.Features[k++] = HashCombine(_HashWs0_Ts0_Tn0, HashCombine(Ws0, Ts0, Tn0));
            buffer.Features[k++] = HashCombine(_HashWs0_Ts0_Wn0_Tn0, HashCombine(Ws0, Ts0, Wn0, Tn0));
            buffer.Features[k++] = HashCombine(_HashTs0_Tn0, HashCombine(Ts0, Tn0));
            buffer.Features[k++] = HashCombine(_HashTn0_Tn1, HashCombine(Tn0, Tn1));

            buffer.Features[k++] = HashCombine(_HashTn0_Tn1_Tn2, HashCombine(Tn0, Tn1, Tn2));
            buffer.Features[k++] = HashCombine(_HashTs0_Tn0_Tn1, HashCombine(Ts0, Tn0, Tn1));
            buffer.Features[k++] = HashCombine(_HashTs0_Ts1_Tn0, HashCombine(Ts0, Ts1, Tn0));
            buffer.Features[k++] = HashCombine(_HashTs0_Ts0f1_Tn0, HashCombine(Ts0, Ts0f1, Tn0));
            buffer.Features[k++] = HashCombine(_HashTs0_Tn0_Tn0b1, HashCombine(Ts0, Tn0, Tn0b1));
            buffer.Features[k++] = HashCombine(_HashTs0_Ts0b1_Ts0b2, HashCombine(Ts0, Ts0b1, Ts0b2));
            buffer.Features[k++] = HashCombine(_HashTs0_Ts0f1_Ts0f2, HashCombine(Ts0, Ts0f1, Ts0f2));
            buffer.Features[k++] = HashCombine(_HashTn0_Tn0b1_Tn0b2, HashCombine(Tn0, Tn0b1, Tn0b2));
            buffer.Features[k++] = HashCombine(_HashTs0_Ts1_Ts1, HashCombine(Ts0, Ts1, Ts1));

            buffer.Features[k++] = HashCombine(_HashWs0_Vs0f, HashCombine(Ws0, Vs0f));
            buffer.Features[k++] = HashCombine(_HashWs0_Vs0b, HashCombine(Ws0, Vs0b));
            buffer.Features[k++] = HashCombine(_HashWn0_Vn0b, HashCombine(Wn0, Vn0b));
            buffer.Features[k++] = HashCombine(_HashTs0_Vs0f, HashCombine(Ts0, Vs0f));
            buffer.Features[k++] = HashCombine(_HashTs0_Vs0b, HashCombine(Ts0, Vs0b));
            buffer.Features[k++] = HashCombine(_HashTn0_Vn0b, HashCombine(Tn0, Vn0b));
            buffer.Features[k++] = HashCombine(_HashWs0_Ds0n0, HashCombine(Ws0, Ds0n0));
            buffer.Features[k++] = HashCombine(_HashWn0_Ds0n0, HashCombine(Wn0, Ds0n0));
            buffer.Features[k++] = HashCombine(_HashTs0_Ds0n0, HashCombine(Ts0, Ds0n0));
            buffer.Features[k++] = HashCombine(_HashTn0_Ds0n0, HashCombine(Tn0, Ds0n0));
            buffer.Features[k++] = HashCombine(_Hash_t_Tn0_Ts0_Ds0n0, HashCombine(Tn0, Ts0, Ds0n0));
            buffer.Features[k++] = HashCombine(_Hash_w_Wn0_Ws0_Ds0n0, HashCombine(Wn0, Ws0, Ds0n0));
        }

        //private static int HashCombine(ReadOnlySpan<char> rhs, long lhs)
        //{
        //    return Hashes.HashCombine(GetHash(rhs), lhs);
        //}

        //private static int HashCombine(string rhs, long lhs)
        //{
        //    return Hashes.HashCombine(GetHash(rhs), lhs);
        //}

        private static int HashCombine(long rhs, long lhs)
        {
            if (lhs == _HashEmpty) { return _HashEmpty; }
            return Hashes.CombineWeak(rhs, lhs);
        }

        private static int HashCombine(long a, long b, long c)
        {
            return HashCombine(a, HashCombine(b, c));
        }

        private static int HashCombine(long a, long b, long c, long d)
        {
            return HashCombine(HashCombine(a, b), HashCombine(c, d));
        }

        private (int a, int b, int c) GetStackContext(int depth, ref LightStack<int> stack, ref int[] data)
        {
            if (depth > 2)
            {
                return (data[stack[0]], data[stack[1]], data[stack[2]]);
            }
            else if (depth > 1)
            {
                return (data[stack[0]], data[stack[1]], _HashEmpty);
            }
            else if (depth == 1)
            {
                return (data[stack[0]], _HashEmpty, _HashEmpty);
            }
            else
            {
                return (_HashEmpty, _HashEmpty, _HashEmpty);
            }
        }

        private (int a, int b, int c) GetBufferContext(int i, int n, ref int[] data)
        {
            if (i == n)
            {
                return (_HashEmpty, _HashEmpty, _HashEmpty);
            }
            if (i >= n - 1)
            {
                return (data[i], _HashEmpty, _HashEmpty);
            }
            else if (i >= n - 2)
            {
                return (data[i], data[i + 1], _HashEmpty);
            }
            else
            {
                return (data[i], data[i + 1], data[i + 2]);
            }
        }

        private (int a, int b, int c) GetParseContext(int i, ref List<List<int>> deps, ref int[] data)
        {
            if (i == -1) { return (0, _HashEmpty, _HashEmpty); }

            var dep = deps[i];
            int valency = dep.Count;

            if (valency < 1) { return (0, _HashEmpty, _HashEmpty); }
            if (valency == 1) { return (1, data[dep.Last()], _HashEmpty); }

            return (2, data[dep.Last()], data[dep[valency - 2]]);
        }

        private bool IsBetween(int target, int from, int to, int[] gold)
        {
            for (int i = from; i < to; i++)
            {
                if (gold[i] == target || gold[target] == i) { return true; }
            }
            return false;
        }

        private bool IsBetween(int target, LightStack<int> others, int[] gold)
        {
            for (int i = 0; i < others.Length; i++)
            {
                int o = others[i];
                if (gold[o] == target || gold[target] == o) { return true; }
            }
            return false;
        }

        private (bool LeftIsGold, bool RightIsGold, bool ShiftIsGold) GetGoldMoves(int i, int N, ref Buffer buffer, int[] golds)
        {
            var (CanLeft, CanRight, CanShift) = GetValidMoves(i, N, buffer.Stack.Length);

            if (buffer.Stack.Length == 0 || (CanShift && golds[i] == buffer.Stack[0])) { return (false, false, true); }

            if (golds[buffer.Stack[0]] == i) { return (true, false, false); }

            //If the word behind s0 is its gold head, Left is incorrect
            if (CanLeft && buffer.Stack.Length > 3 && golds[buffer.Stack[0]] == buffer.Stack[1]) { CanLeft = false; }

            // If there are any dependencies between n0 and the stack pushing n0 will lose them.
            if (CanShift && IsBetween(i, buffer.Stack, golds)) { CanShift = false; }

            // If there are any dependencies between s0 and the buffer, popping s0 will lose them.
            if (CanRight || CanRight) { if (IsBetween(buffer.Stack[0], i + 1, N - 2, golds)) { CanRight = false; CanLeft = false; } }

            //If there are no actions left, do a shift
            if (!(CanLeft || CanRight || CanShift)) { CanShift = true; }

            return (CanLeft, CanRight, CanShift);
        }

        private (bool CanLeft, bool CanRight, bool CanShift) GetValidMoves(int i, int N, int stackSize)
        {
            return ((stackSize > 0), (stackSize > 1), (i < N));
        }

        private int DoMove(int predictedMove, int i, ref Buffer buffer)
        {
            string lbl = ""; // Data.Index2Label[predictedMove];

            switch (predictedMove)
            {
                case LEFT: { buffer.AddArc(i, buffer.Stack.Pop(), lbl); return i; }
                case RIGHT: { int child = buffer.Stack.Pop(); buffer.AddArc(buffer.Stack[0], child, lbl); return i; }
                case SHIFT: { buffer.Stack.Push(i); return i + 1; }
            }
            throw new Exception("Invalid move: " + predictedMove);
        }

        private int GetBestMove(bool CanLeft, bool CanRight, bool CanShift, ref Buffer buffer)
        {
            var best = float.MinValue; int index = -1;
            for (int i = 0; i < N_ACTIONS; i++)
            {
                if ((CanLeft && i == LEFT) || (CanRight && i == RIGHT) || (CanShift && i == SHIFT))
                {
                    if (buffer.Scores[i] > best) { best = buffer.Scores[i]; index = i; }
                }
            }
            return index;
        }

        private (int[] moves, float[] scores) GetBestMoves(bool CanLeft, bool CanRight, bool CanShift, ref Buffer buffer)
        {
            int n = (CanLeft ? 1 : 0) + (CanRight ? 1 : 0) + (CanShift ? 1 : 0);
            var moves = new int[n]; var scores = new float[n]; int k = 0;

            if (CanShift) { moves[k] = SHIFT; scores[k] = buffer.Scores[SHIFT]; k++; }
            if (CanRight) { moves[k] = RIGHT; scores[k] = buffer.Scores[RIGHT]; k++; }
            if (CanLeft) { moves[k] = LEFT; scores[k] = buffer.Scores[LEFT]; k++; }

            return (moves, scores);
        }

        private void UpdateAverages(int epoch, bool final = false)
        {
            foreach (var feature in Data.Weights)
            {
                var weights = AverageWeights.GetOrAdd(feature.Key, k => new float[N_ACTIONS]);

                for (int i = 0; i < N_ACTIONS; i++)
                {
                    weights[i] += feature.Value[i];    //During training, only accumulates the weights
                    if (final) { weights[i] /= (epoch + 1); } // On final step, divide the total sum by the number of epochs sto reach the average
                }
            }
        }

        private void UpdateModel(int correctAction, int predictedAction, ref Buffer buffer, float learningRate)
        {
            if (correctAction == predictedAction) { return; } //nothing to update

            foreach (var feature in buffer.Features)
            {
                if (feature != _HashEmpty)
                {
                    var weights = Data.Weights.GetOrAdd(feature, k => new float[N_ACTIONS]);
                    weights[correctAction] += learningRate;
                    weights[predictedAction] -= learningRate;
                }
            }
        }

        private void ComputeWeights(ref Buffer buffer)
        {
            for (int i = 0; i < N_ACTIONS; i++) { buffer.Scores[i] = 0f; }

            foreach (var feature in buffer.Features)
            {
                if (feature != _HashEmpty)
                {
                    if (Data.Weights.TryGetValue(feature, out var weights))
                    {
                        for (int i = 0; i < N_ACTIONS; i++)
                        {
                            buffer.Scores[i] += weights[i];
                        }
                    }
                }
            }
            //Applies Softmax on the weights
            //double sum = 0;
            //for (int i = 0; i < N_ACTIONS; i++)
            //{
            //    sum += Math.Exp(buffer.Scores[i]);
            //}
            //if(sum > 1e-5)
            //{
            //    for (int i = 0; i < N_ACTIONS; i++)
            //    {
            //        buffer.Scores[i] = (float)(Math.Exp(buffer.Scores[i])/sum);
            //    }
            //}
        }

        public void Process(IDocument document)
        {
            Predict(document);
        }

        public void Predict(IDocument document)
        {
            foreach (var span in document)
            {
                Predict(span);
            }
        }

        public void Predict(ISpan span)
        {
            var buffer = new Buffer(span, N_FEATURES, N_ACTIONS);
            Predict(ref buffer);
            buffer.CopyBackToSpan();
        }

        private static int GetHash(ReadOnlySpan<char> feature)
        {
            return feature.IgnoreCaseHash32();
        }

        private static int GetHash(string feature)
        {
            return feature.IgnoreCaseHash32();
        }

        internal class Buffer : ICloneable
        {
            public int N;
            public int[] Heads;
            public int[] Hashes;
            public int[] TagHashes;
            public int[] SuffixHashes;
            public int[] PrefixHashes;
            public int[] ShapeHashes;
            public int[] NumericHashes;
            public int[] Moves;
            public float[] Scores;
            public float[] ScoresGold;
            public int[] Features;
            public string[] Labels;
            public List<List<int>> Lefts;
            public List<List<int>> Rights;
            public LightStack<int> Stack;
            public int CurrentIndex;

            public ISpan Span;

            private const string ROOT = "<ROOT>";
            private static readonly int _HashRoot = ROOT.IgnoreCaseHash32();
            private const string EOS = "<EOS>";
            private static readonly int _HashEOS = EOS.IgnoreCaseHash32();

            public Buffer(ISpan span, int n_features, int n_moves)
            {
                N = span.TokensCount + 2;
                Span = span;
                Heads = new int[N];
                Hashes = new int[N];
                TagHashes = new int[N];
                SuffixHashes = new int[N];
                PrefixHashes = new int[N];
                ShapeHashes = new int[N];
                NumericHashes = new int[N];
                Labels = new string[N];
                Lefts = new List<List<int>>();
                Rights = new List<List<int>>();
                Moves = new int[n_moves];
                Features = new int[n_features];
                Scores = new float[n_moves];
                Stack = new LightStack<int>();
                for (int i = 0; i < N; i++)
                {
                    if (i == 0)
                    {
                        TagHashes[0] = _HashRoot;
                        Heads[0] = -2;
                        Hashes[0] = TagHashes[0];
                        SuffixHashes[0] = TagHashes[0];
                        PrefixHashes[0] = TagHashes[0];
                        ShapeHashes[0] = TagHashes[0];
                        NumericHashes[0] = TagHashes[0];
                        Labels[0] = ROOT;
                        Lefts.Add(new List<int>());
                        Rights.Add(new List<int>());
                    }
                    else if (i < N - 1)
                    {
                        var tk = span[i - 1];
                        Heads[i] = -1;
                        Labels[i] = tk.DependencyType;
                        TagHashes[i] = tk.POS.ToString().IgnoreCaseHash32();
                        SuffixHashes[i] = GetSuffixHash(tk.ValueAsSpan);
                        PrefixHashes[i] = GetPrefixHash(tk.ValueAsSpan);
                        ShapeHashes[i] = GetShapeHash(tk.ValueAsSpan, true);
                        NumericHashes[i] = tk.Value.Any(c => char.IsNumber(c)).ToString().IgnoreCaseHash32();
                        Hashes[i] = tk.IgnoreCaseHash;
                        Lefts.Add(new List<int>());
                        Rights.Add(new List<int>());
                    }
                    else
                    {
                        TagHashes[N - 1] = _HashEOS;
                        Heads[N - 1] = -2;
                        Hashes[N - 1] = TagHashes[N - 1];
                        SuffixHashes[N - 1] = TagHashes[N - 1];
                        PrefixHashes[N - 1] = TagHashes[N - 1];
                        ShapeHashes[N - 1] = TagHashes[N - 1];
                        NumericHashes[N - 1] = TagHashes[N - 1];
                        Labels[N - 1] = EOS;
                        Lefts.Add(new List<int>());
                        Rights.Add(new List<int>());
                    }
                }

                Lefts.Add(new List<int>());
                Rights.Add(new List<int>());
            }

            public Buffer(Buffer buffer)
            {
                N = buffer.N;
                Span = buffer.Span;
                Heads = buffer.Heads.ToArray();
                Hashes = buffer.Hashes.ToArray();
                TagHashes = buffer.TagHashes.ToArray();
                SuffixHashes = buffer.SuffixHashes.ToArray();
                PrefixHashes = buffer.PrefixHashes.ToArray();
                ShapeHashes = buffer.ShapeHashes.ToArray();
                Labels = buffer.Labels.ToArray();
                Moves = buffer.Moves.ToArray();
                Features = buffer.Features.ToArray();
                Scores = buffer.Scores.ToArray();
                ScoresGold = buffer.Scores.ToArray();
                Stack = buffer.Stack.Clone();
                CurrentIndex = buffer.CurrentIndex;

                Lefts = new List<List<int>>();
                Rights = new List<List<int>>();

                for (int i = 0; i < buffer.Lefts.Count; i++)
                {
                    Lefts.Add(buffer.Lefts[i].ToList());
                }

                for (int i = 0; i < buffer.Rights.Count; i++)
                {
                    Rights.Add(buffer.Rights[i].ToList());
                }
            }

            private static readonly int _H_Base = GetHash("shape");
            private static readonly int _H_Digit = GetHash("shape_digit");
            private static readonly int _H_Lower = GetHash("shape_lower");
            private static readonly int _H_Upper = GetHash("shape_upper");
            private static readonly int _H_Punct = GetHash("shape_puct");
            private static readonly int _H_Symbol = GetHash("shape_symbol");

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static int GetShapeHash(ReadOnlySpan<char> token, bool compact)
            {
                int hash = _H_Base;
                char prev = (char)0;
                for (int i = 0; i < token.Length; i++)
                {
                    if (!compact || token[i] != prev)
                    {
                        prev = token[i];
                        int type;
                        var slice = token.Slice(i, 1);
                        if (char.IsLower(prev)) { type = _H_Lower; }
                        else if (char.IsUpper(prev)) { type = _H_Upper; }
                        else if (char.IsNumber(prev)) { type = _H_Digit; }
                        else if (char.IsPunctuation(prev)) { type = _H_Punct; }
                        else { type = _H_Symbol; }
                        hash = global::UID.Hashes.CombineWeak(hash, type);
                    }
                }
                return hash;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static int GetHash(ReadOnlySpan<char> feature)
            {
                return feature.IgnoreCaseHash32();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static int GetHash(string feature)
            {
                return feature.IgnoreCaseHash32();
            }

            //public static int GetShapeHash(ReadOnlySpan<char> token)
            //{
            //    var sb = new StringBuilder();

            //    //foreach(var c in token)
            //    for(int i = 0; i < token.Length; i++)
            //    {
            //        if (char.IsDigit(token[i]))       { sb.Append("D"); }
            //        else if (char.IsLower(token[i]))  { sb.Append("a"); }
            //        else if (char.IsUpper(token[i]))  { sb.Append("A"); }
            //        else if (char.IsSymbol(token[i])) { sb.Append("#"); }
            //        else                              { sb.Append("?"); }
            //    }

            //    return sb.ToString().GetStableHashCode();
            //}

            private static int GetSuffixHash(ReadOnlySpan<char> token, int suffixSize = 3)
            {
                int len = token.Length - 1;
                int n = Math.Min(suffixSize, len);
                return token.IgnoreCaseHash32(len - n + 1, len);
            }

            internal void CopyBackToSpan()
            {
                for (int i = 1; i < N - 1; i++)
                {
                    if (Heads[i] == N) { Heads[i] = 0; } //the token with Head == N in the buffer is the root
                    Span[i - 1].Head = Heads[i] - 1;
                    Span[i - 1].DependencyType = Labels[i];
                }
            }

            private static int GetPrefixHash(ReadOnlySpan<char> token, int prefixSize = 1)
            {
                int len = token.Length - 1;
                int n = Math.Min(prefixSize, len);
                return token.IgnoreCaseHash32(0, n);
            }

            internal void AddArc(int head, int child, string label)
            {
                Heads[child] = head;
                Labels[child] = label;
                if (child < head)
                {
                    Lefts[head].Add(child);
                }
                else
                {
                    Rights[head].Add(child);
                }
            }

            public object Clone()
            {
                return new Buffer(this);
            }
        }
    }
}