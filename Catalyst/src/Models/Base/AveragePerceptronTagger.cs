using UID;
using Mosaik.Core;

//using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.IO;
using Microsoft.Extensions.Logging;
using MessagePack;

namespace Catalyst.Models
{
    public class AveragePerceptronTaggerModel : StorableObjectData
    {
        public Dictionary<int, float[]> Weights { get; set; } = new Dictionary<int, float[]>();
        public Dictionary<int, int> TokenToSingleTag { get; set; } = new Dictionary<int, int>();

        [IgnoreMember] internal AveragePerceptronTagger.WeightsHolder WeightsHolder { get; set; } = null;
    }

    public class AveragePerceptronTagger : StorableObjectV2<AveragePerceptronTagger, AveragePerceptronTaggerModel>, ITagger, IProcess
    {
        private int N_POS = Enum.GetValues(typeof(PartOfSpeech)).Length;

        private Dictionary<int, float[]> AverageWeights { get; set; }

        public AveragePerceptronTagger(Language language, int version, string tag = "") : base(language, version, tag)
        {
            TagHashes = new int[Enum.GetValues(typeof(PartOfSpeech)).Length];
            TagTagHashes = new int[Enum.GetValues(typeof(PartOfSpeech)).Length][];
            foreach (var pos in Enum.GetValues(typeof(PartOfSpeech)))
            {
                TagHashes[(int)pos] = GetHash(pos.ToString());
                TagTagHashes[(int)pos] = new int[Enum.GetValues(typeof(PartOfSpeech)).Length];
                foreach (var pos2 in Enum.GetValues(typeof(PartOfSpeech)))
                {
                    TagTagHashes[(int)pos][(int)pos2] = Hashes.CombineWeak(TagHashes[(int)pos], GetHash(pos2.ToString()));
                }
            }
        }

        public new static async Task<AveragePerceptronTagger> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new AveragePerceptronTagger(language, version, tag);
            await a.LoadDataAsync();
            a.Data.WeightsHolder ??= new WeightsHolder(a.Data.Weights);
            a.Data.Weights = null;
            return a;
        }

        public override async Task LoadAsync(Stream stream)
        {
            await base.LoadAsync(stream);
            Data.WeightsHolder ??= new WeightsHolder(Data.Weights);
            Data.Weights = null;
        }

        public override async Task StoreAsync(Stream stream)
        {
            if (Data.WeightsHolder is object)
            {
                Data.Weights = Data.WeightsHolder.GetOriginal();
                await base.StoreAsync(stream);
                Data.Weights = null;
            }
            else
            {
                await base.StoreAsync(stream);
            }
        }

        public override async Task StoreAsync()
        {
            if(Data.WeightsHolder is object)
            {
                Data.Weights = Data.WeightsHolder.GetOriginal();
                await base.StoreAsync();
                Data.Weights = null;
            }
            else
            {
                await base.StoreAsync();
            }
        }

        public void Train(IEnumerable<IDocument> documents, int trainingSteps)
        {
            Data.TokenToSingleTag.Clear();
            Data.Weights.Clear();

            AverageWeights = new Dictionary<int, float[]>();

            Data.TokenToSingleTag = ExtractSingleTag(documents);

            var sentences = documents.SelectMany(doc => doc.Spans).ToList();

            var sw = new System.Diagnostics.Stopwatch();

            Span<float> ScoreBuffer = stackalloc float[N_POS];
            Span<int> Features = stackalloc int[N_Features];

            for (int step = 0; step < trainingSteps; step++)
            {
                sentences.Shuffle();
                sw.Restart();
                double total = 0;

                int TP = 0, FN = 0, FP = 0; double precision, recall;
                foreach (var sentence in sentences)
                {
                    var (_TP, _FN, _FP) = TrainOnSentence(sentence, ScoreBuffer, Features);
                    TP += _TP; FN += _FN; FP += _FP;
                    total += sentence.Tokens.Count();
                }
                sw.Stop();

                precision = (double)TP / (TP + FP);
                recall = (double)TP / (TP + FN);

                Logger.LogInformation($"Training {Language} Step {step + 1}/{trainingSteps}: F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% at a rate of {Math.Round(1000 * total / sw.ElapsedMilliseconds, 0) } tokens/second");

                UpdateAverages();
            }

            UpdateAverages(final: true, trainingSteps: trainingSteps);

            Data.Weights = AverageWeights;
            AverageWeights = null;
        }

        private void UpdateAverages(bool final = false, float trainingSteps = -1)
        {
            foreach (var feature in Data.Weights)
            {
                if (!AverageWeights.TryGetValue(feature.Key, out float[] weights))
                {
                    weights = new float[N_POS];
                    AverageWeights.Add(feature.Key, weights);
                }

                for (int i = 0; i < N_POS; i++)
                {
                    weights[i] += feature.Value[i];
                    if (final)
                    {
                        weights[i] /= trainingSteps;
                    }
                }
            }
        }

        public (int TP, int FN, int FP) TrainOnSentence(ISpan span, Span<float> ScoreBuffer, Span<int> features)
        {
            IToken prev = SpecialToken.BeginToken; IToken prev2 = SpecialToken.BeginToken; IToken curr = SpecialToken.BeginToken; IToken next = SpecialToken.BeginToken; IToken next2 = SpecialToken.BeginToken;
            int prevTag = (int)PartOfSpeech.NONE; int prev2Tag = (int)PartOfSpeech.NONE; int currTag = (int)PartOfSpeech.NONE;

            int i = 0, correct = 0;
            int TP = 0, FN = 0, FP = 0;

            var en = span.GetEnumerator();

            while (next != SpecialToken.EndToken)
            {
                prev2 = prev; prev = curr; curr = next; next = next2; prev2Tag = prevTag; prevTag = currTag;
                if (en.MoveNext()) { next2 = en.Current; } else { next2 = SpecialToken.EndToken; }
                if (curr != SpecialToken.BeginToken)
                {
                    int tokenTag = (int)curr.POS;
                    if (!Data.TokenToSingleTag.TryGetValue(curr.IgnoreCaseHash, out currTag))
                    {
                        GetFeatures(features, curr, prev, prev2, next, next2, prevTag, prev2Tag);
                        currTag = PredictTagFromFeatures(features, ScoreBuffer);
                        UpdateModel(tokenTag, currTag, features);
                    }

                    if (currTag == tokenTag) { correct++; }
                    if (currTag == tokenTag) { TP++; }
                    if (currTag != tokenTag) { FP++; FN++; } //Same if we are not evaluating per-tag precision / recall

                    i++;
                }
            }
            return (TP, FN, FP);
        }

        public void Process(IDocument document)
        {
            Predict(document);
        }

        public void Predict(IDocument document)
        {
            Span<float> ScoreBuffer = stackalloc float[N_POS];
            Span<int> Features = stackalloc int[N_Features];
            foreach (var span in document)
            {
                Predict(span, ScoreBuffer, Features);
            }
        }

        public void Predict(ISpan span)
        {
            Span<float> ScoreBuffer = stackalloc float[N_POS];
            Span<int> Features = stackalloc int[N_Features];
            Predict(span, ScoreBuffer, Features);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Predict(ISpan span, Span<float> ScoreBuffer, Span<int> features)
        {
            IToken prev = SpecialToken.BeginToken; IToken prev2 = SpecialToken.BeginToken; IToken curr = SpecialToken.BeginToken; IToken next = SpecialToken.BeginToken; IToken next2 = SpecialToken.BeginToken;
            int prevTag = (int)PartOfSpeech.NONE; int prev2Tag = (int)PartOfSpeech.NONE; int currTag = (int)PartOfSpeech.NONE;

            int i = 0;

            var en = span.GetEnumerator();

            while (next != SpecialToken.EndToken)
            {
                prev2 = prev; prev = curr; curr = next; next = next2; prev2Tag = prevTag; prevTag = currTag;
                if (en.MoveNext()) { next2 = en.Current; } else { next2 = SpecialToken.EndToken; }

                if (curr != SpecialToken.BeginToken)
                {
                    if (!Data.TokenToSingleTag.TryGetValue(curr.IgnoreCaseHash, out int tag))
                    {
                        GetFeatures(features, curr, prev, prev2, next, next2, prevTag, prev2Tag);
                        tag = PredictTagFromFeatures(features, ScoreBuffer);
                    }
                    span.SetTokenTag(i, (PartOfSpeech)tag);
                    currTag = (int)tag;
                    i++;
                }
            }
        }

        private Dictionary<int, int> ExtractSingleTag(IEnumerable<IDocument> documents)
        {
            int minimumFrequency = 20;
            var tokensWithSingleTag = documents.SelectMany(doc => doc.Spans.SelectMany(span => span.Tokens))
                                               .GroupBy(token => token.IgnoreCaseHash)
                                               .Where(g => g.Count() > minimumFrequency)
                                               .Where(g => g.All(token => token.POS == g.First().POS))
                                               .Select(g => g.First())
                                               .ToList();

            var duplicatedHashes = tokensWithSingleTag.GroupBy(token => token.IgnoreCaseHash).Where(g => g.Count() > 1).ToList();
            if (duplicatedHashes.Count > 0)
            {
                duplicatedHashes.ForEach(dh => Console.WriteLine(string.Join(" == ", dh.Select(t => t.Value))));
                throw new Exception("Duplicated hashes found");
            }
            return tokensWithSingleTag.ToDictionary(tk => tk.IgnoreCaseHash, tk => (int)tk.POS);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateModel(int correctTag, int predictedTag, Span<int> features)
        {
            if (correctTag == predictedTag) { return; } //nothing to update
            foreach (var feature in features)
            {
                if (!Data.Weights.TryGetValue(feature, out float[] weights))
                {
                    weights = new float[N_POS];
                    Data.Weights.Add(feature, weights);
                }

                weights[correctTag] += 1f;
                if (predictedTag != (int)PartOfSpeech.NONE) { weights[predictedTag] -= 1f; }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int PredictTagFromFeatures(Span<int> features, Span<float> scoreBuffer)
        {
            bool first = true;

            if (Data.WeightsHolder is object)
            {
                for (int i = 0; i < features.Length; i++)
                {
                    if (Data.WeightsHolder.TryGetValue(features[i], out var weights))
                    {
                        if (first)
                        {
                            weights.CopyTo(scoreBuffer);
                            first = false;
                        }
                        else
                        {
                            for (var j = 0; j < scoreBuffer.Length; j++)
                            {
                                scoreBuffer[j] += weights[j];
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < features.Length; i++)
                {
                    if (Data.Weights.TryGetValue(features[i], out float[] weights))
                    {
                        if (first)
                        {
                            weights.CopyTo(scoreBuffer);
                            first = false;
                        }
                        else
                        {
                            for (var j = 0; j < scoreBuffer.Length; j++)
                            {
                                scoreBuffer[j] += weights[j];
                            }
                        }
                    }
                }
            }

            var best = scoreBuffer[0]; int index = 0;
            for (int i = 1; i < scoreBuffer.Length; i++)
            {
                if (scoreBuffer[i] > best) { best = scoreBuffer[i]; index = i; }
            }

            return best > 0 ? index : (int)PartOfSpeech.NONE;
        }

        private int[] TagHashes;
        private int[][] TagTagHashes;

        private const int N_Features = 32 - 20 + 5 + 5;
        private static readonly int _HashBias              = GetHash("bias");
        private static readonly int _HashIWord             = GetHash("i word");
        private static readonly int _HashIm1Word           = GetHash("i-1 word");
        private static readonly int _HashIm2Word           = GetHash("i-2 word");
        private static readonly int _HashIp1Word           = GetHash("i+1 word");
        private static readonly int _HashIp2Word           = GetHash("i+2 word");
        private static readonly int _HashIShape            = GetHash("i shape");
        private static readonly int _HashIm1Shape          = GetHash("i-1 shape");
        private static readonly int _HashIm2Shape          = GetHash("i-2 shape");
        private static readonly int _HashIp1Shape          = GetHash("i+1 shape");
        private static readonly int _HashIp2Shape          = GetHash("i+2 shape");
        private static readonly int _HashISuf3             = GetHash("i suf3");
        private static readonly int _HashIm1Suf3           = GetHash("i-1 suf3");
        private static readonly int _HashIm2Suf3           = GetHash("i-2 suf3");
        private static readonly int _HashIp1Suf3           = GetHash("i+1 suf3");
        private static readonly int _HashIp2Suf3           = GetHash("i+2 suf3");
        private static readonly int _HashTagIm1            = GetHash("i-1 tag");
        private static readonly int _HashTagIm2            = GetHash("i-2 tag");
        private static readonly int _HashTagIm1WordI       = GetHash("i-1 tag i word");
        private static readonly int _HashTagIm2WordI       = GetHash("i-2 tag i word");
        private static readonly int _HashTagIm2TagIm1      = GetHash("i-2 tag i-1 tag");
        private static readonly int _HashTagIm2TagIm1WordI = GetHash("i-2 tag i-1 tag i word");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GetFeatures(Span<int> features, IToken current, IToken prev, IToken prev2, IToken next, IToken next2, int prevTag, int prev2Tag)
        {
            int k = 0;

            features[k++] = HashCombine(_HashBias, _HashBias);
            features[k++] = HashCombine(_HashIWord, current.IgnoreCaseHash);
            features[k++] = HashCombine(_HashIm1Word, prev.IgnoreCaseHash);
            features[k++] = HashCombine(_HashIm2Word, prev2.IgnoreCaseHash);
            features[k++] = HashCombine(_HashIp1Word, next.IgnoreCaseHash);
            features[k++] = HashCombine(_HashIp2Word, next2.IgnoreCaseHash);

            features[k++] = HashCombine(_HashIShape, GetShapeHash(current.ValueAsSpan, true));
            features[k++] = HashCombine(_HashIm1Shape, GetShapeHash(prev.ValueAsSpan, true));
            features[k++] = HashCombine(_HashIm2Shape, GetShapeHash(prev2.ValueAsSpan, true));
            features[k++] = HashCombine(_HashIp1Shape, GetShapeHash(next.ValueAsSpan, true));
            features[k++] = HashCombine(_HashIp2Shape, GetShapeHash(next2.ValueAsSpan, true));
            features[k++] = HashCombine(_HashISuf3, GetSuffixHash(current.ValueAsSpan, 3));
            features[k++] = HashCombine(_HashIm1Suf3, GetSuffixHash(prev.ValueAsSpan, 3));
            features[k++] = HashCombine(_HashIm2Suf3, GetSuffixHash(prev2.ValueAsSpan, 3));
            features[k++] = HashCombine(_HashIp1Suf3, GetSuffixHash(next.ValueAsSpan, 3));
            features[k++] = HashCombine(_HashIp2Suf3, GetSuffixHash(next2.ValueAsSpan, 3));
            features[k++] = HashCombine(_HashTagIm1, TagHashes[prevTag]);
            features[k++] = HashCombine(_HashTagIm2, TagHashes[prev2Tag]);
            features[k++] = HashCombine(_HashTagIm1WordI, HashCombine(TagHashes[prevTag], current.IgnoreCaseHash));
            features[k++] = HashCombine(_HashTagIm2WordI, HashCombine(TagHashes[prev2Tag], current.IgnoreCaseHash));
            features[k++] = HashCombine(_HashTagIm2TagIm1, TagTagHashes[prev2Tag][prevTag]);
            features[k++] = HashCombine(_HashTagIm2TagIm1WordI, HashCombine(TagTagHashes[prev2Tag][prevTag], current.IgnoreCaseHash));
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
                    hash = Hashes.CombineWeak(hash, type);
                }
            }
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetSuffixHash(ReadOnlySpan<char> token, int suffixSize = 3)
        {
            int len = token.Length - 1;
            int n = Math.Min(suffixSize, len);
            return token.IgnoreCaseHash32(len - n + 1, len);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetPrefixHash(ReadOnlySpan<char> token, int prefixSize = 1)
        {
            int len = token.Length - 1;
            int n = Math.Min(prefixSize, len);
            return token.IgnoreCaseHash32(0, n);
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

        private static int HashCombine(long rhs, long lhs)
        {
            return Hashes.CombineWeak(rhs, lhs);
        }


        internal sealed class WeightsHolder
        {
            private readonly float[] _weights;
            private float[] _zero;
            private readonly int _singleWeightLength;
            private readonly Dictionary<int, int> _positions;
            private readonly int _maxIndex;

            public WeightsHolder(Dictionary<int, float[]> weights)
            {
                _weights = new float[weights.Values.Sum(v => v.Count(f => f != 0f))];
                _singleWeightLength = weights.First().Value.Length;
                
                _positions = new Dictionary<int, int>(weights.Count);
                var ws = _weights.AsSpan();
                int curPos = 0;
                int maxIndex = 0;
                foreach(var kv in weights)
                {
                    if (kv.Value.Any(v => v != 0f))
                    {
                        _positions.Add(kv.Key, curPos);
                        kv.Value.AsSpan().CopyTo(ws.Slice(curPos));
                        curPos += kv.Value.Length;
                    }
                    maxIndex = Math.Max(kv.Key, maxIndex);
                }
                _maxIndex = maxIndex;
            }

            public bool TryGetValue(int index, out ReadOnlySpan<float> weights)
            {
                if(_positions.TryGetValue(index, out var start))
                {
                    weights = _weights.AsSpan(start, _singleWeightLength);
                    return true;
                }
                else if(index <= _maxIndex)
                {
                    //Empty value
                    if(_zero is null)
                    {
                        _zero = new float[_singleWeightLength];
                    }

                    weights = _zero;

                    return true;
                }
                weights = default;
                return false;
            }

            internal Dictionary<int, float[]> GetOriginal()
            {
                var dict = new Dictionary<int, float[]>();

                for(int i = 0; i <= _maxIndex; i++)
                {
                    if (!_positions.ContainsKey(i))
                    {
                        dict[i] = new float[_singleWeightLength];
                    }
                }

                foreach(var kv in _positions)
                {
                    dict[kv.Key] = _weights.AsSpan(kv.Value, _singleWeightLength).ToArray();
                }

                return dict;
            }
        }
    }
}