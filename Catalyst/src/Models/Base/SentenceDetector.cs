using UID;
using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class SentenceDetectorModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public Dictionary<int, float[]> Weights { get; set; }
    }

    public class SentenceDetector : StorableObject<SentenceDetector, SentenceDetectorModel>, ISentenceDetector, IProcess
    {
        private const int N_Tags = 2;
        private Dictionary<int, float[]> AverageWeights { get; set; }
        private int[] HashLengths;

        public SentenceDetector(Language language, int version, string tag = "") : base(language, version, tag)
        {
            Tokenizer = new FastTokenizer(language);
            HashLengths = Enumerable.Range(0, 100).Select(i => $"HashLength{i}".IgnoreCaseHash32()).ToArray();
        }

        public new static async Task<SentenceDetector> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new SentenceDetector(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        private FastTokenizer Tokenizer;

        public void Process(IDocument document)
        {
            Parse(document);
        }

        public void Parse(IDocument document)
        {
            if (document.Length == 0) { return; }

            if (document.Spans.Count() != 1)
            {
                throw new InvalidOperationException("Document must be tokenized first");
            }

            var tokens = document.Spans.First().Tokens.ToList();

            bool hasReplacements = false;
            //NOTE: This loop is not used for anything here, but instead to force tokens to cache the replacement
            //      As they'll not be able to retrieve it later when re-added to the document.
            for (int i = 0; i < tokens.Count; i++)
            {
                hasReplacements |= (tokens[i].Replacement is null);
            }

            //var tokens = SentenceDetectorTokenizer(document.Value).ToList();
            var text = document.Value;

            //TODO: FIX THIS NOT TO USE STRING; USE SPAN INSTEAD

            const int padding = 2;
            var paddedTokens = new List<IToken>(tokens.Count + 2 * padding);

            paddedTokens.Add(SpecialToken.BeginToken);
            paddedTokens.Add(SpecialToken.BeginToken);
            paddedTokens.AddRange(tokens);
            paddedTokens.Add(SpecialToken.EndToken);
            paddedTokens.Add(SpecialToken.EndToken);

            int N = paddedTokens.Count;

            var isSentenceEnd = new bool[N];
            for (int i = padding + 1; i < N - padding - 1; i++) //Skip BeginTokens and EndTokens, and first and last token of sentence
            {
                if (paddedTokens[i].ValueAsSpan.IsSentencePunctuation())
                {
                    var features = GetFeatures(paddedTokens, i);
                    isSentenceEnd[i] = PredictTagFromFeatures(features, Data.Weights);
                }
            }

            document.Clear();

            //Now split the original document at the right places
            var separators = CharacterClasses.WhitespaceCharacters;

            //If any sentence detected within the single span (i.e. ignoring the first and last tokens
            if (isSentenceEnd.AsSpan().Slice(padding + 1, tokens.Count - 1).IndexOf(true) >= 0)
            {
                int offset = 0;
                for (int i = padding; i < N - padding; i++)
                {
                    if (isSentenceEnd[i])
                    {
                        int b = offset;
                        int e = tokens[i - padding].End;
                        if (e < b) { continue; }
                        while (char.IsWhiteSpace(text[b]) && b < e) { b++; }

                        while (char.IsWhiteSpace(text[e]) && e > b) { e--; }

                        try
                        {
                            if (!text.AsSpan().Slice(b, e - b + 1).IsNullOrWhiteSpace())
                            {
                                var span = document.AddSpan(b, e);
                                foreach (var t in tokens)
                                {
                                    if (t.Begin >= span.Begin && t.End <= span.End)
                                    {
                                        span.AddToken(t); //Re-add the tokens back in the document
                                    }
                                }
                            }
                        }
                        catch (Exception)
                        {
                            Logger.LogCritical("Failed to tokenize: b={b} e={e} l={l} offset={offset} tEnd={tEnd} i={i} tCount={tCount}", b, e, text.Length, offset, tokens[i - padding].End, i, tokens.Count);
                            throw;
                        }
                        offset = e + 1;
                    }
                }
                if (offset <= document.Length - 1)
                {
                    int b = offset;
                    int e = document.Length - 1;
                    while (char.IsWhiteSpace(text[b]) && b < e) { b++; }

                    while (char.IsWhiteSpace(text[e]) && e > b) { e--; }

                    if (!text.AsSpan().Slice(b, e - b + 1).IsNullOrWhiteSpace())
                    {
                        var span = document.AddSpan(b, e);
                        foreach (var t in tokens)
                        {
                            if (t.Begin >= span.Begin && t.End <= span.End)
                            {
                                span.AddToken(t);
                            }
                        }
                    }
                }
            }
            else
            {
                int b = 0;
                int e = document.Length - 1;
                while (char.IsWhiteSpace(text[b]) && b < e) { b++; }
                while (char.IsWhiteSpace(text[e]) && e > b) { e--; }

                var span = document.AddSpan(b, e);
                foreach (var t in tokens)
                {
                    if (t.Begin >= span.Begin && t.End <= span.End)
                    {
                        span.AddToken(t); //Re-add the tokens back in the document
                    }
                }
            }
        }

        public IEnumerable<IToken> SentenceDetectorTokenizer(string input)
        {
            return Tokenizer.Parse(input);
        }

        public void Train(List<List<SentenceDetectorToken>> sentences, int trainingSteps = 20)
        {
            Data = new SentenceDetectorModel();
            Data.Weights = new Dictionary<int, float[]>();
            AverageWeights = new Dictionary<int, float[]>();

            var sw = new System.Diagnostics.Stopwatch();
            var rng = new Random();

            for (int step = 0; step < trainingSteps; step++)
            {
                sentences.Shuffle();

                sw.Start();

                double correct = 0, total = 0, totalTokens = 0; bool first = true;
                var filteredSentences = sentences.Where(s => s.Count > 5 && s.Last().IsPunctuation == true);

                foreach (var tokens in filteredSentences)
                {
                    var paddedTokens = new List<string>();
                    var isSentenceEnd = new List<bool>();
                    if (rng.NextDouble() > 0.1)
                    {
                        var tmp = Enumerable.Reverse(sentences.RandomItem()).Take(2).Reverse();
                        paddedTokens.AddRange(tmp.Select(st => st.Value));
                        isSentenceEnd.AddRange(tmp.Select(st => st.IsSentenceEnd));
                    }
                    else
                    {
                        paddedTokens.Add(SpecialToken.BOS); paddedTokens.Add(SpecialToken.BOS);
                        isSentenceEnd.Add(false); isSentenceEnd.Add(false);
                    }

                    paddedTokens.AddRange(tokens.Select(st => st.Value));
                    isSentenceEnd.AddRange(tokens.Select(st => st.IsSentenceEnd));

                    if (rng.NextDouble() > 0.1)
                    {
                        var tmp = sentences.RandomItem().Take(2);
                        paddedTokens.AddRange(tmp.Select(st => st.Value));
                        isSentenceEnd.AddRange(tmp.Select(st => st.IsSentenceEnd));
                    }
                    else
                    {
                        paddedTokens.Add(SpecialToken.EOS); paddedTokens.Add(SpecialToken.EOS);
                        isSentenceEnd.Add(false); isSentenceEnd.Add(false);
                    }

                    correct += TrainOnSentence(paddedTokens, isSentenceEnd, ref first); ;

                    total += tokens.Count(tk => tk.IsSentenceEnd); ;

                    totalTokens += tokens.Count;
                }
                sw.Stop();
                Console.WriteLine($"{Languages.EnumToCode(Language)} Step {step + 1}/{trainingSteps}: {Math.Round(100 * correct / total, 2)}% at a rate of {Math.Round(1000 * totalTokens / sw.ElapsedMilliseconds, 0) } tokens/second");
                sw.Restart();

                UpdateAverages();
            }

            UpdateAverages(final: true, trainingSteps: trainingSteps);

            FinishTraining();
        }

        private void FinishTraining()
        {
            Data.Weights = AverageWeights;
            Data.TrainedTime = DateTime.UtcNow;

            AverageWeights = null;
            TrainingTokenMemory = null;
            TrainingGuessMemory = null;
        }

        private void UpdateAverages(bool final = false, int trainingSteps = -1)
        {
            foreach (var featureDict in Data.Weights)
            {
                if (!AverageWeights.TryGetValue(featureDict.Key, out float[] weights))
                {
                    weights = new float[N_Tags];
                    AverageWeights.Add(featureDict.Key, weights);
                }
                var featureValues = featureDict.Value.ToArray();
                weights[0] += featureValues[0];
                weights[1] += featureValues[1];
                if (final)
                {
                    weights[0] /= trainingSteps;
                    weights[1] /= trainingSteps;
                }
            }
        }

        private IToken[] TrainingTokenMemory = new IToken[0];
        private bool[] TrainingGuessMemory = new bool[0];

        public int TrainOnSentence(List<string> sentenceTokensWithPadding, List<bool> IsSentenceEnd, ref bool isFirst)
        {
            int correct = 0;
            int N = sentenceTokensWithPadding.Count;
            const int padding = 2;

            while ((N + 2 * padding) > TrainingTokenMemory.Length)
            {
                TrainingTokenMemory = new IToken[TrainingTokenMemory.Length * 2 + 100];
                TrainingGuessMemory = new bool[TrainingTokenMemory.Length];
            }

            for (int i = 0; i < N; i++)
            {
                TrainingTokenMemory[i] = new FakeToken(sentenceTokensWithPadding[i]);
            }

            for (int i = padding; i < N - padding; i++) //Skip BeginTokens and EndTokens
            {
                var features = GetFeatures(TrainingTokenMemory, i);
                TrainingGuessMemory[i] = PredictTagFromFeatures(features, Data.Weights);
                UpdateModel(IsSentenceEnd[i], TrainingGuessMemory[i], features);
                if (IsSentenceEnd[i] && IsSentenceEnd[i] == TrainingGuessMemory[i]) { correct++; }
            }

            return correct;
        }

        private void UpdateModel(bool correctTag, bool predictedTag, int[] features)
        {
            if (correctTag == predictedTag) { return; } //nothing to update
            foreach (var feature in features)
            {
                if (!Data.Weights.TryGetValue(feature, out float[] weights))
                {
                    weights = new float[N_Tags];
                    Data.Weights.Add(feature, weights);
                }

                weights[correctTag ? 1 : 0] += 1f;
                weights[predictedTag ? 1 : 0] -= 1f;
            }
        }

        private bool PredictTagFromFeatures(int[] features, Dictionary<int, float[]> weightsSource)
        {
            float[] scoreDict = new float[N_Tags];

            foreach (var feature in features)
            {
                if (weightsSource.TryGetValue(feature, out float[] weights))
                {
                    scoreDict[0] += weights[0];
                    scoreDict[1] += weights[1];
                }
            }
            return scoreDict[1] >= scoreDict[0];
        }

        private readonly int _Hash_Bias = GetHash("_HashBias");
        private readonly int _Hash_True_IIsPunct = GetHash("TRUE_HashIIsPunct");
        private readonly int _Hash_True_Im1IsCap = GetHash("TRUE_HashIm1IsCap");
        private readonly int _Hash_True_Ip1IsCap = GetHash("TRUE_HashIp1IsCap");
        private readonly int _Hash_True_Im1Upp = GetHash("TRUE_HashIm1Upp");
        private readonly int _Hash_True_Ip1Upp = GetHash("TRUE_HashIp1Upp");
        private readonly int _Hash_True_Im1Low = GetHash("TRUE_HashIm1Low");
        private readonly int _Hash_True_Ip1Low = GetHash("TRUE_HashIp1Low");
        private readonly int _Hash_True_IIsPunctIp1IsPunct = GetHash("TRUE_HashIIsPunctIp1IsPunct");
        private readonly int _Hash_True_IIsPunctIm1IsPunct = GetHash("TRUE_HashIIsPunctIm1IsPunct");
        private readonly int _Hash_True_IIsPunctIp2IsPunct = GetHash("TRUE_HashIIsPunctIp2IsPunct");
        private readonly int _Hash_True_IIsPunctIm2IsPunct = GetHash("TRUE_HashIIsPunctIm2IsPunct");
        private readonly int _Hash_True_IIsCurrency = GetHash("TRUE_HashIIsCurrency");
        private readonly int _Hash_True_IIsNumeric = GetHash("TRUE_HashIIsNumeric");
        private readonly int _Hash_True_IHasNumeric = GetHash("TRUE_HashIHasNumeric");
        private readonly int _Hash_True_IsPunctIp1Quote = GetHash("TRUE_HashIsPunctIp1Quote");
        private readonly int _Hash_True_IequalIm1 = GetHash("TRUE_HashIequalIm1");
        private readonly int _Hash_True_IequalIm2 = GetHash("TRUE_HashIequalIm2");
        private readonly int _Hash_True_IequalIp1 = GetHash("TRUE_HashIequalIp1");
        private readonly int _Hash_True_IequalIp2 = GetHash("TRUE_HashIequalIp2");
        private readonly int _Hash_True_Im1IsBOS = GetHash("TRUE_HashIm1IsBOS");
        private readonly int _Hash_True_Im2IsBOS = GetHash("TRUE_HashIm2IsBOS");
        private readonly int _Hash_True_Ip1IsEOS = GetHash("TRUE_HashIp1IsEOS");
        private readonly int _Hash_True_Ip2IsEOS = GetHash("TRUE_HashIp2IsEOS");

        private readonly int _Hash_False_IIsPunct = GetHash("FALSE_HashIIsPunct");
        private readonly int _Hash_False_Im1IsCap = GetHash("FALSE_HashIm1IsCap");
        private readonly int _Hash_False_Ip1IsCap = GetHash("FALSE_HashIp1IsCap");
        private readonly int _Hash_False_Im1Upp = GetHash("FALSE_HashIm1Upp");
        private readonly int _Hash_False_Ip1Upp = GetHash("FALSE_HashIp1Upp");
        private readonly int _Hash_False_Im1Low = GetHash("FALSE_HashIm1Low");
        private readonly int _Hash_False_Ip1Low = GetHash("FALSE_HashIp1Low");
        private readonly int _Hash_False_IIsPunctIp1IsPunct = GetHash("FALSE_HashIIsPunctIp1IsPunct");
        private readonly int _Hash_False_IIsPunctIm1IsPunct = GetHash("FALSE_HashIIsPunctIm1IsPunct");
        private readonly int _Hash_False_IIsPunctIp2IsPunct = GetHash("FALSE_HashIIsPunctIp2IsPunct");
        private readonly int _Hash_False_IIsPunctIm2IsPunct = GetHash("FALSE_HashIIsPunctIm2IsPunct");
        private readonly int _Hash_False_IIsCurrency = GetHash("FALSE_HashIIsCurrency");
        private readonly int _Hash_False_IIsNumeric = GetHash("FALSE_HashIIsNumeric");
        private readonly int _Hash_False_IHasNumeric = GetHash("FALSE_HashIHasNumeric");
        private readonly int _Hash_False_IsPunctIp1Quote = GetHash("FALSE_HashIsPunctIp1Quote");
        private readonly int _Hash_False_IequalIm1 = GetHash("FALSE_HashIequalIm1");
        private readonly int _Hash_False_IequalIm2 = GetHash("FALSE_HashIequalIm2");
        private readonly int _Hash_False_IequalIp1 = GetHash("FALSE_HashIequalIp1");
        private readonly int _Hash_False_IequalIp2 = GetHash("FALSE_HashIequalIp2");
        private readonly int _Hash_False_Im1IsBOS = GetHash("FALSE_HashIm1IsBOS");
        private readonly int _Hash_False_Im2IsBOS = GetHash("FALSE_HashIm2IsBOS");
        private readonly int _Hash_False_Ip1IsEOS = GetHash("FALSE_HashIp1IsEOS");
        private readonly int _Hash_False_Ip2IsEOS = GetHash("FALSE_HashIp2IsEOS");

        private readonly int _Hash_FirstChar = GetHash("Hash_FirstChar");
        private readonly int _Hash_Im1Length = GetHash("Hash_Im1Length");
        private readonly int _Hash_Ip1Length = GetHash("Hash_Ip1Length");

        public int[] GetFeatures(IList<IToken> tokens, int indexCurrent)
        {
            var current = tokens[indexCurrent];
            var prev2 = tokens[indexCurrent - 2];
            var prev = tokens[indexCurrent - 1];
            var next = tokens[indexCurrent + 1];
            var next2 = tokens[indexCurrent + 2];

            //Features inspired by iSentenizer, but extended for better results (https://www.hindawi.com/journals/tswj/2014/196574/)
            var features = new int[27];
            features[0] = _Hash_Bias;// (GetHash(("bias")));

            features[1] = (current.ValueAsSpan.IsSentencePunctuation()) ? _Hash_True_IIsPunct : _Hash_False_IIsPunct;
            features[2] = (prev.ValueAsSpan.IsCapitalized()) ? _Hash_True_Im1IsCap : _Hash_False_Im1IsCap;
            features[3] = (next.ValueAsSpan.IsCapitalized()) ? _Hash_True_Ip1IsCap : _Hash_False_Ip1IsCap;
            features[4] = (prev.ValueAsSpan.IsAllUpperCase()) ? _Hash_True_Im1Upp : _Hash_False_Im1Upp;
            features[5] = (next.ValueAsSpan.IsAllUpperCase()) ? _Hash_True_Ip1Upp : _Hash_False_Ip1Upp;
            features[6] = (prev.ValueAsSpan.IsAllLowerCase()) ? _Hash_True_Im1Low : _Hash_False_Im1Low;
            features[7] = (next.ValueAsSpan.IsAllLowerCase()) ? _Hash_True_Ip1Low : _Hash_False_Ip1Low;
            features[8] = (current.ValueAsSpan.IsSentencePunctuation() && next.ValueAsSpan.IsSentencePunctuation()) ? _Hash_True_IIsPunctIp1IsPunct : _Hash_False_IIsPunctIp1IsPunct;
            features[9] = (current.ValueAsSpan.IsSentencePunctuation() && prev.ValueAsSpan.IsSentencePunctuation()) ? _Hash_True_IIsPunctIm1IsPunct : _Hash_False_IIsPunctIm1IsPunct;
            features[10] = (current.ValueAsSpan.IsSentencePunctuation() && next2.ValueAsSpan.IsSentencePunctuation()) ? _Hash_True_IIsPunctIp2IsPunct : _Hash_False_IIsPunctIp2IsPunct;
            features[11] = (current.ValueAsSpan.IsSentencePunctuation() && prev2.ValueAsSpan.IsSentencePunctuation()) ? _Hash_True_IIsPunctIm2IsPunct : _Hash_False_IIsPunctIm2IsPunct;
            features[12] = (current.ValueAsSpan.IsCurrency()) ? _Hash_True_IIsCurrency : _Hash_False_IIsCurrency;
            features[13] = (current.ValueAsSpan.IsNumeric()) ? _Hash_True_IIsNumeric : _Hash_False_IIsNumeric;
            features[14] = (current.ValueAsSpan.HasNumeric()) ? _Hash_True_IHasNumeric : _Hash_False_IHasNumeric;
            features[15] = (current.ValueAsSpan.IsSentencePunctuation() && next.ValueAsSpan.IsOpenQuote()) ? _Hash_True_IsPunctIp1Quote : _Hash_False_IsPunctIp1Quote;
            features[16] = (current.ValueAsSpan.IsSentencePunctuation() && current == prev) ? _Hash_True_IequalIm1 : _Hash_False_IequalIm1;
            features[17] = (current.ValueAsSpan.IsSentencePunctuation() && current == prev2) ? _Hash_True_IequalIm2 : _Hash_False_IequalIm2;
            features[18] = (current.ValueAsSpan.IsSentencePunctuation() && current == next) ? _Hash_True_IequalIp1 : _Hash_False_IequalIp1;
            features[19] = (current.ValueAsSpan.IsSentencePunctuation() && current == next2) ? _Hash_True_IequalIp2 : _Hash_False_IequalIp2;
            features[20] = (prev.Length == 0 && prev.Value == SpecialToken.BOS) ? _Hash_True_Im1IsBOS : _Hash_False_Im1IsBOS;
            features[21] = (next.Length == 0 && next.Value == SpecialToken.BOS) ? _Hash_True_Im2IsBOS : _Hash_False_Im2IsBOS;
            features[22] = (prev.Length == 0 && prev.Value == SpecialToken.EOS) ? _Hash_True_Ip1IsEOS : _Hash_False_Ip1IsEOS;
            features[23] = (next.Length == 0 && next.Value == SpecialToken.EOS) ? _Hash_True_Ip2IsEOS : _Hash_False_Ip2IsEOS;
            features[24] = Hashes.CombineWeak(_Hash_FirstChar, current.Length > 0 ? current.ValueAsSpan.IgnoreCaseHash32(0, 0) : 0);
            features[25] = Hashes.CombineWeak(_Hash_Im1Length, HashLengths[Math.Min(99, prev.Length)]);
            features[26] = Hashes.CombineWeak(_Hash_Ip1Length, HashLengths[Math.Min(99, next.Length)]);

            return features;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetHash(string feature)
        {
            return feature.IgnoreCaseHash32();
        }

        private struct FakeToken : IToken
        {
            public FakeToken(string value) : this()
            {
                Value = value;
            }

            public int Begin { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public int End { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public int Length => Value.Length;

            public int Index => throw new NotImplementedException();

            public string Value { get; set; }

            public ReadOnlySpan<char> ValueAsSpan => Value.AsSpan();

            public string Replacement { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public int Hash { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public int IgnoreCaseHash { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public Dictionary<string, string> Metadata => throw new NotImplementedException();

            public PartOfSpeech POS { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public EntityType[] EntityTypes => throw new NotImplementedException();

            public int Head { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public string DependencyType { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
            public float Frequency { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

            public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

            public void ClearEntities() => throw new NotImplementedException();

            public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

            public void RemoveEntityType(int ix) => throw new NotImplementedException();

            public void RemoveEntityType(string entityType) => throw new NotImplementedException();
        }

        public class SentenceDetectorToken
        {
            public string Value { get; private set; }
            public int Length { get; private set; }
            public int Begin { get; private set; }
            public int End { get; private set; }
            public bool IsSentenceEnd { get; set; } = false;
            public bool IsPunctuation { get { return Value.AsSpan().IsSentencePunctuation(); } }

            public SentenceDetectorToken(string value, int begin, int end)
            {
                //Better to cache once all the flags to avoid having to recompute them during training
                Value = value;
                Begin = begin;
                End = end;
                Length = Value.Length;
            }

            public override string ToString()
            {
                return Value;
            }
        }
    }
}