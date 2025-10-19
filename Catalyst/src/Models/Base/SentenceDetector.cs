using UID;
using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Threading;
using System.Runtime.InteropServices;
using System.Buffers;

namespace Catalyst.Models
{
    public class SentenceDetectorModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public Dictionary<int, float[]> Weights { get; set; }
    }

    public class SentenceDetector : StorableObjectV2<SentenceDetector, SentenceDetectorModel>, ISentenceDetector, IProcess
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

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Parse(document, cancellationToken);
        }

        public void Parse(IDocument document, CancellationToken cancellationToken = default)
        {
            if (document.Length == 0) { return; }

            if (document.SpansCount != 1)
            {
                return; //Document has already been tokenized and passed to the sentence detection, so ignore the second call
            }

            var rentedTokens = document.Spans.First().ToTokenSpanPolled(out var actualLength);
            var tokens = rentedTokens.AsSpan(0, actualLength);

            if (tokens.Length == 0) 
            {
                ArrayPool<Token>.Shared.Return(rentedTokens);
                return; 
            }

            int[] rentedBegins = tokens.Length < 256 ? null : ArrayPool<int>.Shared.Rent(tokens.Length);
            int[] rentedEnds   = tokens.Length < 256 ? null : ArrayPool<int>.Shared.Rent(tokens.Length);

            Span<int> tokensBegins = tokens.Length < 256 ? stackalloc int[tokens.Length] : rentedBegins.AsSpan(0,tokens.Length);
            Span<int> tokensEnds   = tokens.Length < 256 ? stackalloc int[tokens.Length] : rentedEnds.AsSpan(0, tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                tokensBegins[i] = tokens[i].Begin;
                tokensEnds[i]   = tokens[i].End;
            }

            bool hasReplacements = false;
            
            //NOTE: This loop is not used for anything here, but instead to force tokens to cache the replacement
            //      As they'll not be able to retrieve it later when re-added to the document.
            for (int i = 0; i < tokens.Length; i++)
            {
                hasReplacements |= (tokens[i].Replacement is null);
            }

            var text = document.Value.AsSpan();

            const int padding = 2;

            int N = tokens.Length + 2 * padding;

            var rentedPaddedTokens = ArrayPool<Token>.Shared.Rent(N);
            var paddedTokens = rentedPaddedTokens.AsSpan(0,N);

            paddedTokens[0] = Token.BeginToken;
            paddedTokens[1] = Token.BeginToken;

            for(int i = 0; i < tokens.Length; i++)
            {
                paddedTokens[i + 2] = tokens[i];
            }

            paddedTokens[paddedTokens.Length - 2] = Token.EndToken;
            paddedTokens[paddedTokens.Length - 1] = Token.EndToken;

            var rentedIsSentenceEnd = ArrayPool<bool>.Shared.Rent(N);

            var isSentenceEnd = rentedIsSentenceEnd.AsSpan(0, N);
            isSentenceEnd.Fill(false);

            Span<int> features = stackalloc int[27];

            for (int i = padding + 1; i < N - padding - 1; i++) //Skip BeginTokens and EndTokens, and first and last token of sentence
            {
                if (paddedTokens[i].ValueAsSpan.IsSentencePunctuation())
                {
                    GetFeatures(paddedTokens, i, features);
                    isSentenceEnd[i] = PredictTagFromFeatures(features, Data.Weights);
                }
                cancellationToken.ThrowIfCancellationRequested();
            }

            document.Clear();

            //Now split the original document at the right places
            
            //If any sentence detected within the single span (i.e. ignoring the first and last tokens
            if (isSentenceEnd.Slice(padding + 1, tokens.Length - 1).IndexOf(true) >= 0)
            {
                int offset = 0;
                int lastBegin = 0;
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
                            if (!text.Slice(b, e - b + 1).IsNullOrWhiteSpace())
                            {
                                var span = document.AddSpan(b, e);
                                var spanBegin = span.Begin;
                                var spanEnd   = span.End;
                                for (int itoken = lastBegin; itoken < tokens.Length; itoken++)
                                {
                                    var tb = tokensBegins[itoken];
                                    if (tb >= spanBegin)
                                    {
                                        var te = tokensEnds[itoken];
                                        if (te <= spanEnd)
                                        {
                                            ref Token t = ref tokens[itoken];
                                            span.AddToken(t); //Re-add the tokens back in the document
                                            lastBegin = itoken;
                                        }
                                        else
                                        {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception)
                        {
                            Logger.LogCritical("Failed to tokenize: b={b} e={e} l={l} offset={offset} tEnd={tEnd} i={i} tCount={tCount}", b, e, text.Length, offset, tokens[i - padding].End, i, tokens.Length);
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

                    if (!text.Slice(b, e - b + 1).IsNullOrWhiteSpace())
                    {
                        var span = document.AddSpan(b, e);
                        var spanBegin = span.Begin;
                        var spanEnd = span.End;
                        for (int itoken = lastBegin; itoken < tokens.Length; itoken++)
                        {
                            var tb = tokensBegins[itoken];
                            if (tb >= spanBegin)
                            {
                                var te = tokensEnds[itoken];
                                if (te <= spanEnd)
                                {
                                    ref Token t = ref tokens[itoken];
                                    span.AddToken(t); //Re-add the tokens back in the document
                                    lastBegin = itoken;
                                }
                                else
                                {
                                    break;
                                }
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
                var spanBegin = span.Begin;
                var spanEnd = span.End;
                for (int itoken = 0; itoken < tokens.Length; itoken++)
                {
                    var tb = tokensBegins[itoken];
                    if (tb >= spanBegin)
                    {
                        var te = tokensEnds[itoken];
                        if (te <= spanEnd)
                        {
                            ref Token t = ref tokens[itoken];
                            span.AddToken(t); //Re-add the tokens back in the document
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            if (rentedBegins is object)
            {
                ArrayPool<int>.Shared.Return(rentedBegins);
                ArrayPool<int>.Shared.Return(rentedEnds);
            }

            ArrayPool<bool>.Shared.Return(rentedIsSentenceEnd);
            ArrayPool<Token>.Shared.Return(rentedTokens);
            ArrayPool<Token>.Shared.Return(rentedPaddedTokens);
        }

        public IEnumerable<IToken> SentenceDetectorTokenizer(string input)
        {
            return Tokenizer.Parse(input);
        }

        public double Train(List<List<SentenceDetectorToken>> sentences, int trainingSteps = 20)
        {
            Data = new SentenceDetectorModel();
            Data.Weights = new Dictionary<int, float[]>();
            AverageWeights = new Dictionary<int, float[]>();

            var sw = new System.Diagnostics.Stopwatch();
            var rng = new Random();
            
            double precision = 0;

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
                        paddedTokens.Add(Token.BOS); paddedTokens.Add(Token.BOS);
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
                        paddedTokens.Add(Token.EOS); paddedTokens.Add(Token.EOS);
                        isSentenceEnd.Add(false); isSentenceEnd.Add(false);
                    }

                    correct += TrainOnSentence(paddedTokens, isSentenceEnd, ref first); ;

                    total += tokens.Count(tk => tk.IsSentenceEnd); ;

                    totalTokens += tokens.Count;
                }

                precision = correct / total;
                sw.Stop();
                Logger.LogInformation($"{Languages.EnumToCode(Language)} Step {step + 1}/{trainingSteps}: {Math.Round(100 * correct / total, 2)}% at a rate of {Math.Round(1000 * totalTokens / sw.ElapsedMilliseconds, 0) } tokens/second");
                sw.Restart();

                UpdateAverages();
            }

            UpdateAverages(final: true, trainingSteps: trainingSteps);

            FinishTraining();
            return precision * 100;
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

        private Token[] TrainingTokenMemory = null;
        private bool[]  TrainingGuessMemory = null;

        public int TrainOnSentence(List<string> sentenceTokensWithPadding, List<bool> IsSentenceEnd, ref bool isFirst)
        {
            int correct = 0;
            int N = sentenceTokensWithPadding.Count;
            const int padding = 2;

            TrainingTokenMemory ??= new Token[0];
            TrainingGuessMemory ??= new bool[0];

            while ((N + 2 * padding) > TrainingTokenMemory.Length)
            {
                TrainingTokenMemory = new Token[TrainingTokenMemory.Length * 2 + 100];
                TrainingGuessMemory = new bool[TrainingTokenMemory.Length];
            }

            for (int i = 0; i < N; i++)
            {
                TrainingTokenMemory[i] = Token.Fake(sentenceTokensWithPadding[i]);
            }

            Span<int> features = stackalloc int[27];

            for (int i = padding; i < N - padding; i++) //Skip BeginTokens and EndTokens
            {
                GetFeatures(TrainingTokenMemory, i, features);
                TrainingGuessMemory[i] = PredictTagFromFeatures(features, Data.Weights);
                UpdateModel(IsSentenceEnd[i], TrainingGuessMemory[i], features);
                if (IsSentenceEnd[i] && IsSentenceEnd[i] == TrainingGuessMemory[i]) { correct++; }
            }

            return correct;
        }

        private void UpdateModel(bool correctTag, bool predictedTag, ReadOnlySpan<int> features)
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

        private bool PredictTagFromFeatures(ReadOnlySpan<int> features, Dictionary<int, float[]> weightsSource)
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

#if NET5_0_OR_GREATER
        internal void GetFeatures(Span<Token> tokens, int indexCurrent, Span<int> features)
        {
            ref Token current = ref tokens[indexCurrent];
            ref Token prev2   = ref tokens[indexCurrent - 2];
            ref Token prev    = ref tokens[indexCurrent - 1];
            ref Token next    = ref tokens[indexCurrent + 1];
            ref Token next2   = ref tokens[indexCurrent + 2];

            //Features inspired by iSentenizer, but extended for better results (https://www.hindawi.com/journals/tswj/2014/196574/)
            
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
            features[16] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == prev.Value) ? _Hash_True_IequalIm1 : _Hash_False_IequalIm1;
            features[17] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == prev2.Value) ? _Hash_True_IequalIm2 : _Hash_False_IequalIm2;
            features[18] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == next.Value) ? _Hash_True_IequalIp1 : _Hash_False_IequalIp1;
            features[19] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == next2.Value) ? _Hash_True_IequalIp2 : _Hash_False_IequalIp2;
            features[20] = (prev.Length == 0 && prev.Value == Token.BOS) ? _Hash_True_Im1IsBOS : _Hash_False_Im1IsBOS;
            features[21] = (next.Length == 0 && next.Value == Token.BOS) ? _Hash_True_Im2IsBOS : _Hash_False_Im2IsBOS;
            features[22] = (prev.Length == 0 && prev.Value == Token.EOS) ? _Hash_True_Ip1IsEOS : _Hash_False_Ip1IsEOS;
            features[23] = (next.Length == 0 && next.Value == Token.EOS) ? _Hash_True_Ip2IsEOS : _Hash_False_Ip2IsEOS;
            features[24] = Hashes.CombineWeak(_Hash_FirstChar, current.Length > 0 ? current.ValueAsSpan.IgnoreCaseHash32(0, 0) : 0);
            features[25] = Hashes.CombineWeak(_Hash_Im1Length, HashLengths[Math.Min(99, prev.Length)]);
            features[26] = Hashes.CombineWeak(_Hash_Ip1Length, HashLengths[Math.Min(99, next.Length)]);
        }

#else

        internal void GetFeatures(Span<Token> tokens, int indexCurrent, Span<int> features)
        {
            Token current = tokens[indexCurrent];
            Token prev2   = tokens[indexCurrent - 2];
            Token prev    = tokens[indexCurrent - 1];
            Token next    = tokens[indexCurrent + 1];
            Token next2   = tokens[indexCurrent + 2];

            //Features inspired by iSentenizer, but extended for better results (https://www.hindawi.com/journals/tswj/2014/196574/)
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
            features[16] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == prev.Value) ? _Hash_True_IequalIm1 : _Hash_False_IequalIm1;
            features[17] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == prev2.Value) ? _Hash_True_IequalIm2 : _Hash_False_IequalIm2;
            features[18] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == next.Value) ? _Hash_True_IequalIp1 : _Hash_False_IequalIp1;
            features[19] = (current.ValueAsSpan.IsSentencePunctuation() && current.Value == next2.Value) ? _Hash_True_IequalIp2 : _Hash_False_IequalIp2;
            features[20] = (prev.Length == 0 && prev.Value == Token.BOS) ? _Hash_True_Im1IsBOS : _Hash_False_Im1IsBOS;
            features[21] = (next.Length == 0 && next.Value == Token.BOS) ? _Hash_True_Im2IsBOS : _Hash_False_Im2IsBOS;
            features[22] = (prev.Length == 0 && prev.Value == Token.EOS) ? _Hash_True_Ip1IsEOS : _Hash_False_Ip1IsEOS;
            features[23] = (next.Length == 0 && next.Value == Token.EOS) ? _Hash_True_Ip2IsEOS : _Hash_False_Ip2IsEOS;
            features[24] = Hashes.CombineWeak(_Hash_FirstChar, current.Length > 0 ? current.ValueAsSpan.IgnoreCaseHash32(0, 0) : 0);
            features[25] = Hashes.CombineWeak(_Hash_Im1Length, HashLengths[Math.Min(99, prev.Length)]);
            features[26] = Hashes.CombineWeak(_Hash_Ip1Length, HashLengths[Math.Min(99, next.Length)]);
        }
#endif

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int GetHash(string feature)
        {
            return feature.IgnoreCaseHash32();
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