using UID;
using Mosaik.Core;

//using MessagePack;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;

namespace Catalyst.Models
{
    public class AveragePerceptronEntityRecognizerModel : StorableObjectData
    {
        public DateTime TrainedTime { get; set; }

        public Dictionary<int, string> IndexToEntityType { get; set; }
        public Dictionary<int, EntityTag> IndexToEntityTag { get; set; }

        public ConcurrentDictionary<int, float[]> Weights { get; set; }

        public string[] Tags { get; set; }
        public int[] TagHashes { get; set; }
        public int[][] TagTagHashes { get; set; }

        public List<HashSet<int>> Gazeteers { get; set; }
        public string[] EntityTypes { get; set; }

        public bool IgnoreCase { get; set; }
    }

    public class AveragePerceptronEntityRecognizer : StorableObject<AveragePerceptronEntityRecognizer, AveragePerceptronEntityRecognizerModel>, IEntityRecognizer, IProcess
    {
        private int N_Features = 21;
        private int N_Tags;

        private int[] POShashes;

        private ConcurrentDictionary<int, float[]> AverageWeights { get; set; }

        public Dictionary<string, int> MapEntityTypeToTag { get; set; }

        public bool IgnoreCase { get { return Data.IgnoreCase; } set { Data.IgnoreCase = value; } }

        private AveragePerceptronEntityRecognizer(Language language, int version, string tag) : base(language, version, tag, compress: true)
        {
            POShashes = new int[Enum.GetValues(typeof(PartOfSpeech)).Length];
            foreach (var pos in Enum.GetValues(typeof(PartOfSpeech)))
            {
                POShashes[(int)pos] = GetHash(pos.ToString());
            }
        }

        public AveragePerceptronEntityRecognizer(Language language, int version, string tag, string[] entityTypes = null, bool ignoreCase = false) : this(language, version, tag)
        {
            if (entityTypes is object)
            {
                Data = new AveragePerceptronEntityRecognizerModel();
                InitializeEntityTypes(entityTypes);
                N_Tags = Data.Tags.Length;
            }
            Data.IgnoreCase = ignoreCase;
        }

        private void InitializeEntityTypes(string[] entityTypes)
        {
            Data.EntityTypes = entityTypes;
            Data.Tags = new string[entityTypes.Length * 4 + 1];
            Data.IndexToEntityType = new Dictionary<int, string>();
            Data.IndexToEntityTag = new Dictionary<int, EntityTag>();

            //TagOutside must be the first in the tag list, as it's the default tag in the indexing (a.k.a IndexTagOutside)
            Data.Tags[0] = TagOutside.ToString();

            int i = 1;
            foreach (var et in entityTypes)
            {
                foreach (var s in new EntityTag[] { EntityTag.Begin, EntityTag.Inside, EntityTag.End, EntityTag.Single })
                {
                    Data.Tags[i] = $"{(char)s}{Separator}{et}";
                    Data.IndexToEntityType.Add(i, et);
                    Data.IndexToEntityTag.Add(i, s);
                    i++;
                }
            }

            int N = Data.Tags.Length;
            Data.TagHashes = new int[N];
            Data.TagTagHashes = new int[N][];

            MapEntityTypeToTag = new Dictionary<string, int>();

            for (i = 0; i < N; i++)
            {
                Data.TagHashes[i] = GetHash(Data.Tags[i]);
                Data.TagTagHashes[i] = new int[N];
                for (int j = 0; j < N; j++)
                {
                    Data.TagTagHashes[i][j] = Hashes.CombineWeak(Data.TagHashes[i], GetHash(Data.Tags[j]));
                }

                MapEntityTypeToTag.Add(Data.Tags[i], i);
            }
        }

        public const char TagBegin = (char)EntityTag.Begin;
        public const char TagInside = (char)EntityTag.Inside;
        public const char TagEnd = (char)EntityTag.End; //Last
        public const char TagOutside = (char)EntityTag.Outside;
        public const char TagSingle = (char)EntityTag.Single; //Unit
        public const string Separator = "_";
        private const int IndexTagOutside = 0;

        public new static async Task<AveragePerceptronEntityRecognizer> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new AveragePerceptronEntityRecognizer(language, version, tag);
            await a.LoadDataAsync();
            a.N_Tags = a.Data.Tags.Length;
            a.N_Features += 3 * a.Data.Gazeteers.Count;
            return a;
        }

        public string SingleOrOutside(IList<EntityType> types)
        {
            EntityType tmp = types.Where(et => Data.EntityTypes.Any(t => t == et.Type)).FirstOrDefault();

            if (tmp.Type != null)
            {
                return $"{(char)tmp.Tag}{Separator}{tmp.Type}";
            }
            else
            {
                return TagOutside.ToString();
            }
        }

        public void Train(IEnumerable<IDocument> documents, int trainingSteps = 10, List<List<string>> gazeteers = null)
        {
            if (gazeteers is object)
            {
                Data.Gazeteers = gazeteers.Select(l => new HashSet<int>(l.Select(s => GetHash(s)))).ToList();
                N_Features += 3 * Data.Gazeteers.Count;
            }
            else
            {
                Data.Gazeteers = new List<HashSet<int>>();
            }

            Data.Weights = new ConcurrentDictionary<int, float[]>();
            AverageWeights = new ConcurrentDictionary<int, float[]>();

            var sentences = documents.SelectMany(doc => doc.Spans).ToList();

            var sw = new System.Diagnostics.Stopwatch();

            int N_dev = (int)Math.Floor(0.9 * sentences.Count());

            var trainSentences = sentences.Take(N_dev).ToList();
            var testSentences = sentences.Skip(N_dev).ToList();

            var trainSentencesTags = trainSentences.Select(st => st.Tokens.Select(tk => MapEntityTypeToTag[SingleOrOutside(tk.EntityTypes)]).ToArray()).ToArray();
            var testSentencesTags  = testSentences.Select(st => st.Tokens.Select(tk => MapEntityTypeToTag[SingleOrOutside(tk.EntityTypes)]).ToArray()).ToArray();

            string tOutside   = TagOutside.ToString();
            double totalTrain = trainSentences.Sum(st => st.Tokens.Count(tk => SingleOrOutside(tk.EntityTypes) != tOutside));
            double totalTest  = testSentences.Sum(st => st.Tokens.Count(tk => SingleOrOutside(tk.EntityTypes) != tOutside));

            double totalTokensTrain = trainSentences.Sum(st => st.TokensCount);
            double totalTokensTest  = testSentences.Sum(st => st.TokensCount);

            int TP = 0, FN = 0, FP = 0; double precision, recall;

            for (int step = 0; step < trainingSteps; step++)
            {
                trainSentences.ShuffleTogether(trainSentencesTags);

                sw.Restart();

                Parallel.ForEach(Enumerable.Range(0, trainSentences.Count), i =>
                {
                    Span<float> ScoreBuffer = stackalloc float[N_Tags];
                    Span<int> Features      = stackalloc int[N_Features];

                    var (_TP, _FN, _FP) = TrainOnSentence(trainSentences[i], ref trainSentencesTags[i], ScoreBuffer, Features);
                    Interlocked.Add(ref TP, _TP);
                    Interlocked.Add(ref FP, _FP);
                    Interlocked.Add(ref FN, _FN);
                });
                sw.Stop();
                precision = (double)TP / (TP + FP);
                recall = (double)TP / (TP + FN);
                Logger.LogInformation($"{Languages.EnumToCode(Language)} Step {step + 1}/{trainingSteps} Train set: F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% at a rate of {Math.Round(1000 * totalTokensTrain / sw.ElapsedMilliseconds, 0) } tokens/second");

                TP = 0; FN = 0; FP = 0;
                sw.Restart();

                Parallel.ForEach(Enumerable.Range(0, testSentences.Count), i =>
                {
                    Span<float> ScoreBuffer = stackalloc float[N_Tags];
                    Span<int> Features      = stackalloc int[N_Features];

                    var (_TP, _FN, _FP) = TrainOnSentence(testSentences[i], ref testSentencesTags[i], ScoreBuffer, Features, updateModel: false);
                    Interlocked.Add(ref TP, _TP);
                    Interlocked.Add(ref FP, _FP);
                    Interlocked.Add(ref FN, _FN);
                });

                sw.Stop();
                precision = (double)TP / (TP + FP);
                recall    = (double)TP / (TP + FN);
                Logger.LogInformation($"{Languages.EnumToCode(Language)} Step {step + 1}/{trainingSteps} Test set: F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% at a rate of {Math.Round(1000 * totalTokensTest / sw.ElapsedMilliseconds, 0) } tokens/second");

                UpdateAverages();
            }

            UpdateAverages(final: true, trainingSteps: trainingSteps);

            Data.Weights = AverageWeights;
            Data.TrainedTime = DateTime.UtcNow;
            AverageWeights = null;

            //Final test
            TP = 0; FN = 0; FP = 0;
            sw.Restart();
            Parallel.ForEach(Enumerable.Range(0, trainSentences.Count), i =>
            {
                Span<float> ScoreBuffer = stackalloc float[N_Tags];
                Span<int> Features = stackalloc int[N_Features];
                var tags = trainSentencesTags[i];
                var (_TP, _FN, _FP) = TrainOnSentence(trainSentences[i], ref tags, ScoreBuffer, Features, updateModel: false);
                Interlocked.Add(ref TP, _TP);
                Interlocked.Add(ref FP, _FP);
                Interlocked.Add(ref FN, _FN);
            });
            sw.Stop();

            precision = (double)TP / (TP + FP);
            recall = (double)TP / (TP + FN);
            Logger.LogInformation($"{Languages.EnumToCode(Language)} FINAL Train set: F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% at a rate of {Math.Round(1000 * totalTokensTrain / sw.ElapsedMilliseconds, 0) } tokens/second");

            TP = 0; FN = 0; FP = 0;
            sw.Restart();
            Parallel.ForEach(Enumerable.Range(0, testSentences.Count), i =>
            {
                Span<float> ScoreBuffer = stackalloc float[N_Tags];
                Span<int> Features = stackalloc int[N_Features];
                var tags = testSentencesTags[i];
                var (_TP, _FN, _FP) = TrainOnSentence(testSentences[i], ref tags, ScoreBuffer, Features, updateModel: false);
                Interlocked.Add(ref TP, _TP);
                Interlocked.Add(ref FP, _FP);
                Interlocked.Add(ref FN, _FN);
            });
            sw.Stop();
            precision = (double)TP / (TP + FP);
            recall = (double)TP / (TP + FN);
            Logger.LogInformation($"{Languages.EnumToCode(Language)} FINAL Test set: F1={100 * 2 * (precision * recall) / (precision + recall):0.00}% P={100 * precision:0.00}% R={100 * recall:0.00}% at a rate of {Math.Round(1000 * totalTokensTest / sw.ElapsedMilliseconds, 0) } tokens/second");
        }

        private void UpdateAverages(bool final = false, float trainingSteps = -1)
        {
            foreach (var feature in Data.Weights)
            {
                var weights = AverageWeights.GetOrAdd(feature.Key, k => new float[N_Tags]);

                for (int i = 0; i < N_Tags; i++)
                {
                    weights[i] += feature.Value[i];
                    if (final)
                    {
                        weights[i] /= trainingSteps;
                    }
                }
            }
        }

        public (int TP, int FN, int FP) TrainOnSentence(ISpan span, ref int[] spanTags, Span<float> ScoreBuffer, Span<int> features, bool updateModel = true)
        {
            //for training, we expect the tokens to have [BILOU]-[Type] entries as the only EntityType
            IToken prev = SpecialToken.BeginToken; IToken prev2 = SpecialToken.BeginToken; IToken curr = SpecialToken.BeginToken; IToken next = SpecialToken.BeginToken; IToken next2 = SpecialToken.BeginToken;
            int prevTag = IndexTagOutside; int prev2Tag = IndexTagOutside; int currTag = IndexTagOutside;

            int i = 0, correct = 0;

            int TP = 0, FN = 0, FP = 0;

            var en = span.GetEnumerator();

            while (next != SpecialToken.EndToken)
            {
                prev2 = prev; prev = curr; curr = next; next = next2; prev2Tag = prevTag; prevTag = currTag;
                if (en.MoveNext()) { next2 = en.Current; } else { next2 = SpecialToken.EndToken; }
                if (!(curr is SpecialToken))
                {
                    int tokenTag = spanTags[i];

                    GetFeatures(features, curr, prev, prev2, next, next2, prevTag, prev2Tag);

                    currTag = PredictTagFromFeatures(features, ScoreBuffer);

                    if (updateModel) { UpdateModel(tokenTag, currTag, features); }

                    if (tokenTag != IndexTagOutside && currTag == tokenTag) { correct++; }

                    if (tokenTag != IndexTagOutside && currTag == tokenTag) { TP++; }
                    if (tokenTag == IndexTagOutside && currTag != tokenTag) { FP++; }
                    if (tokenTag != IndexTagOutside && currTag != tokenTag) { FN++; }
                    i++;
                }
            }

            return (TP, FN, FP);
        }

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return Data.EntityTypes;
        }

        public bool RecognizeEntities(IDocument document)
        {
            if (N_Tags == 0) { N_Tags = Data.Tags.Length; }
            Span<float> ScoreBuffer = stackalloc float[N_Tags];
            Span<int> Features = stackalloc int[N_Features];
            var result = false;
            foreach (var span in document)
            {
                result |= Predict(span, ScoreBuffer, Features);
            }
            return result;
        }

        public bool Predict(ISpan span)
        {
            Span<float> ScoreBuffer = stackalloc float[N_Tags];
            Span<int> Features = stackalloc int[N_Features];
            return Predict(span, ScoreBuffer, Features);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Predict(ISpan span, Span<float> ScoreBuffer, Span<int> features)
        {
            IToken prev = SpecialToken.BeginToken; IToken prev2 = SpecialToken.BeginToken; IToken curr = SpecialToken.BeginToken; IToken next = SpecialToken.BeginToken; IToken next2 = SpecialToken.BeginToken;
            int prevTag = IndexTagOutside; int prev2Tag = IndexTagOutside; int currTag = IndexTagOutside;
            bool foundAny = false;
            int i = 0;

            var en = span.GetEnumerator();

            var tags = new int[span.TokensCount];

            while (next != SpecialToken.EndToken)
            {
                prev2 = prev; prev = curr; curr = next; next = next2; prev2Tag = prevTag; prevTag = currTag;
                if (en.MoveNext()) { next2 = en.Current; } else { next2 = SpecialToken.EndToken; }

                if (curr != SpecialToken.BeginToken)
                {
                    GetFeatures(features, curr, prev, prev2, next, next2, prevTag, prev2Tag);
                    tags[i] = PredictTagFromFeatures(features, ScoreBuffer);
                    currTag = tags[i];
                    i++;
                }
            }

            string lastBegin = null;

            for (i = 0; i < span.TokensCount; i++)
            {
                if (tags[i] != IndexTagOutside)
                {
                    var type = Data.IndexToEntityType[tags[i]];
                    var tag = Data.IndexToEntityTag[tags[i]];

                    bool valid = tag == EntityTag.Single; //Single is always valid

                    if (tag == EntityTag.Begin) //Checks if it's a valid combination of tags - i.e. B+I+E or B+E
                    {
                        for (int j = i + 1; j < span.TokensCount; j++)
                        {
                            var other_tag = Data.IndexToEntityTag[tags[i]];

                            if (other_tag != EntityTag.Inside || other_tag != EntityTag.End) { break; }

                            var other_type = Data.IndexToEntityType[tags[i]];

                            if (other_type != type) { break; }

                            if (other_tag == EntityTag.End) { valid = true; break; } //found the right tag and right type by now
                        }
                    }
                    else if (tag == EntityTag.Inside || tag == EntityTag.End)
                    {
                        valid = type == lastBegin;
                    }

                    if (valid)
                    {
                        if (tag == EntityTag.Begin) { lastBegin = type; }
                        if (tag == EntityTag.End) { lastBegin = null; }

                        span[i].AddEntityType(new EntityType(type, tag));
                        foundAny = true;
                    }
                }
            }
            return foundAny;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateModel(int correctTag, int predictedTag, Span<int> features)
        {
            if (correctTag == predictedTag) { return; } //nothing to update
            foreach (var feature in features)
            {
                var weights = Data.Weights.GetOrAdd(feature, k => new float[N_Tags]);

                weights[correctTag] += 1f;
                weights[predictedTag] -= 1f;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int PredictTagFromFeatures(Span<int> features, Span<float> ScoreBuffer)
        {
            bool first = true;

            foreach (var feature in features)
            {
                if (Data.Weights.TryGetValue(feature, out float[] weights))
                {
                    if (first)
                    {
                        weights.CopyTo(ScoreBuffer);
                        first = false;
                    }
                    else
                    {
                        for (var j = 0; j < ScoreBuffer.Length; j++)
                        {
                            ScoreBuffer[j] += weights[j];
                        }
                    }
                }
            }

            var best = ScoreBuffer[0]; int index = 0;
            for (int i = 1; i < ScoreBuffer.Length; i++)
            {
                if (ScoreBuffer[i] > best) { best = ScoreBuffer[i]; index = i; }
            }

            return best > 0 ? index : IndexTagOutside;
        }

        private readonly int _HashBias = GetIgnoreCaseHash("bias");
        private readonly int _HashISufix = GetIgnoreCaseHash("i suffix");
        private readonly int _HashIPrefix = GetIgnoreCaseHash("i pref1");
        private readonly int _HashIShape = GetIgnoreCaseHash("i shape");
        private readonly int _HashIm1Sufix = GetIgnoreCaseHash("i-1 suffix");
        private readonly int _HashIp1Sufix = GetIgnoreCaseHash("i+1 suffix");
        private readonly int _HashIm1Shape = GetIgnoreCaseHash("i-1 shape");
        private readonly int _HashIp1Shape = GetIgnoreCaseHash("i+1 shape");
        private readonly int _HashIm1TagIword = GetIgnoreCaseHash("i-1 tag i word");
        private readonly int _HashIm2Word = GetIgnoreCaseHash("i-2 word");
        private readonly int _HashIp1Word = GetIgnoreCaseHash("i+1 word");
        private readonly int _HashIWord = GetIgnoreCaseHash("i word");
        private readonly int _HashIm1Word = GetIgnoreCaseHash("i-1 word");
        private readonly int _HashIp2Word = GetIgnoreCaseHash("i+2 word");
        private readonly int _HashIm1Tag = GetIgnoreCaseHash("i-1 tag");
        private readonly int _HashIm2Tag = GetIgnoreCaseHash("i-2 tag");
        private readonly int _HashITagIm2Tag = GetIgnoreCaseHash("i tag i-2 tag");

        private readonly int _HashIPOS = GetIgnoreCaseHash("i pos");
        private readonly int _HashIm1POS = GetIgnoreCaseHash("i-1 pos");
        private readonly int _HashIm2POS = GetIgnoreCaseHash("i-2 pos");
        private readonly int _HashIp1POS = GetIgnoreCaseHash("i+1 pos");
        private readonly int _HashIp2POS = GetIgnoreCaseHash("i+2 pos");

        private readonly int _HashGazeteerI = GetIgnoreCaseHash("i gazeteer");
        private readonly int _HashGazeteerIm1 = GetIgnoreCaseHash("i-1 gazeteer");
        private readonly int _HashGazeteerIp1 = GetIgnoreCaseHash("i+1 gazeteer");
        private readonly int _HashGazeteerTrue = GetIgnoreCaseHash("gazeteer true");
        private readonly int _HashGazeteerFalse = GetIgnoreCaseHash("gazeteer false");

        public void GetFeatures(IToken[] tokens, int[] guesses, int indexCurrent, Span<int> featuresBuffer)
        {
            var current = tokens[indexCurrent];
            var prev = tokens[indexCurrent - 1];
            var prev2 = tokens[indexCurrent - 2];
            var next = tokens[indexCurrent + 1];
            var next2 = tokens[indexCurrent + 2];

            var prevTag = guesses[indexCurrent - 1];
            var prev2Tag = guesses[indexCurrent - 2];
            GetFeatures(featuresBuffer, current, prev, prev2, next, next2, prevTag, prev2Tag);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void GetFeatures(Span<int> features, IToken current, IToken prev, IToken prev2, IToken next, IToken next2, int prevTag, int prev2Tag)
        {
            //Features from Spacy
            int k = 0;
            bool ignoreCase = Data.IgnoreCase;
            features[k++] = _HashBias;
            features[k++] = Hashes.CombineWeak(_HashISufix, GetSuffixHash(current.ValueAsSpan));
            features[k++] = Hashes.CombineWeak(_HashIPrefix, GetPrefixHash(current.ValueAsSpan));
            features[k++] = Hashes.CombineWeak(_HashIm1Sufix, GetSuffixHash(prev.ValueAsSpan));
            features[k++] = Hashes.CombineWeak(_HashIp1Sufix, GetSuffixHash(next.ValueAsSpan));
            features[k++] = Hashes.CombineWeak(_HashIm1TagIword, Hashes.CombineWeak(Data.TagHashes[prevTag], ignoreCase ? current.IgnoreCaseHash : current.Hash));
            features[k++] = Hashes.CombineWeak(_HashIm2Word, ignoreCase ? prev2.IgnoreCaseHash : prev2.Hash);
            features[k++] = Hashes.CombineWeak(_HashIp1Word, ignoreCase ? next.IgnoreCaseHash : next.Hash);
            features[k++] = Hashes.CombineWeak(_HashIWord, ignoreCase ? current.IgnoreCaseHash : current.Hash);
            features[k++] = Hashes.CombineWeak(_HashIm1Word, ignoreCase ? prev.IgnoreCaseHash : prev.Hash);
            features[k++] = Hashes.CombineWeak(_HashIp2Word, ignoreCase ? next2.IgnoreCaseHash : next2.Hash);
            features[k++] = Hashes.CombineWeak(_HashIm1Tag, Data.TagHashes[prevTag]);
            features[k++] = Hashes.CombineWeak(_HashIm2Tag, Data.TagHashes[prev2Tag]);
            features[k++] = Hashes.CombineWeak(_HashITagIm2Tag, Data.TagTagHashes[prevTag][prev2Tag]);

            features[k++] = Hashes.CombineWeak(_HashIPOS, POShashes[(int)current.POS]);
            features[k++] = Hashes.CombineWeak(_HashIm2POS, POShashes[(int)prev2.POS]);
            features[k++] = Hashes.CombineWeak(_HashIp1POS, POShashes[(int)next.POS]);
            features[k++] = Hashes.CombineWeak(_HashIp2POS, POShashes[(int)next2.POS]);

            features[k++] = Hashes.CombineWeak(_HashIShape, GetShapeHash(current.ValueAsSpan, false));
            features[k++] = Hashes.CombineWeak(_HashIm1Shape, GetShapeHash(current.ValueAsSpan, false));
            features[k++] = Hashes.CombineWeak(_HashIp1Shape, GetShapeHash(current.ValueAsSpan, false));

            for (int i = 0; i < Data.Gazeteers.Count; i++)
            {
                features[k++] = Hashes.CombineWeak(_HashGazeteerI + i, Data.Gazeteers[i].Contains(ignoreCase ? current.IgnoreCaseHash : current.Hash) ? _HashGazeteerTrue : _HashGazeteerFalse);
                features[k++] = Hashes.CombineWeak(_HashGazeteerIm1 + i, Data.Gazeteers[i].Contains(ignoreCase ? prev.IgnoreCaseHash : prev.Hash) ? _HashGazeteerTrue : _HashGazeteerFalse);
                features[k++] = Hashes.CombineWeak(_HashGazeteerIp1 + i, Data.Gazeteers[i].Contains(ignoreCase ? next.IgnoreCaseHash : next.Hash) ? _HashGazeteerTrue : _HashGazeteerFalse);
            }
        }

        private static readonly int _H_Base = GetIgnoreCaseHash("shape");
        private static readonly int _H_Digit = GetIgnoreCaseHash("shape_digit");
        private static readonly int _H_Lower = GetIgnoreCaseHash("shape_lower");
        private static readonly int _H_Upper = GetIgnoreCaseHash("shape_upper");
        private static readonly int _H_Punct = GetIgnoreCaseHash("shape_puct");
        private static readonly int _H_Symbol = GetIgnoreCaseHash("shape_symbol");

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetShapeHash(ReadOnlySpan<char> token, bool compact)
        {
            int hash = _H_Base;
            int prevType = _H_Base;
            for (int i = 0; i < token.Length; i++)
            {
                int type;
                if (char.IsLower(token[i])) { type = _H_Lower; }
                else if (char.IsUpper(token[i])) { type = _H_Upper; }
                else if (char.IsNumber(token[i])) { type = _H_Digit; }
                else if (char.IsPunctuation(token[i])) { type = _H_Punct; }
                else { type = _H_Symbol; }

                if (!compact || type != prevType)
                {
                    hash = Hashes.CombineWeak(hash, type);
                }
                prevType = type;
            }
            return hash;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int GetSuffixHash(ReadOnlySpan<char> token, int suffixSize = 3)
        {
            int len = token.Length - 1;
            int n = Math.Min(suffixSize, len);
            return Data.IgnoreCase ? token.IgnoreCaseHash32(len - n + 1, len) : token.CaseSensitiveHash32(len - n + 1, len);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int GetPrefixHash(ReadOnlySpan<char> token, int prefixSize = 1)
        {
            int len = token.Length - 1;
            int n = Math.Min(prefixSize, len);
            return Data.IgnoreCase ? token.IgnoreCaseHash32(0, n) : token.CaseSensitiveHash32(0, n);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetIgnoreCaseHash(string feature)
        {
            return feature.IgnoreCaseHash32();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int GetHash(string feature)
        {
            return Data.IgnoreCase ? feature.IgnoreCaseHash32() : feature.CaseSensitiveHash32();
        }
    }
}