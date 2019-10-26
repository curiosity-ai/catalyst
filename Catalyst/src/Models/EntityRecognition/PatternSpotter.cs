using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class PatternSpotterModel : StorableObjectData
    {
        public string CaptureTag { get; set; }
        public List<MatchingPattern> Patterns { get; set; } = new List<MatchingPattern>();
    }

    public class PatternSpotter : StorableObject<PatternSpotter, PatternSpotterModel>, IEntityRecognizer, IProcess
    {
        private PatternSpotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public PatternSpotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        public new static async Task<PatternSpotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new PatternSpotter(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { Data.CaptureTag };
        }

        public bool HasAnyEntity(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span, stopOnFirstFound: true);
                if (foundAny) { return true; }
            }
            return false;
        }

        public bool RecognizeEntities(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span, stopOnFirstFound: false);
            }
            return foundAny;
        }

        public void ClearModel()
        {
            Data.Patterns.Clear();
        }

        public bool RecognizeEntities(ISpan ispan, bool stopOnFirstFound = false)
        {
            var tokens = ispan.ToTokenSpan();
            int N = tokens.Length;
            bool foundAny = false;

            for (int i = 0; i < N; i++)
            {
                foreach (var p in Data.Patterns)
                {
                    if (p.IsMatch(tokens.Slice(i), out var consumedTokens))
                    {
                        if (stopOnFirstFound) { return true; } //Do not capture the entity, just returns true to say there's a match

                        if (consumedTokens == 1)
                        {
                            tokens[i].AddEntityType(new EntityType(Data.CaptureTag, EntityTag.Single));
                        }
                        else
                        {
                            for (int j = i; j < (i + consumedTokens); j++)
                            {
                                tokens[j].AddEntityType(new EntityType(Data.CaptureTag, (j == i ? EntityTag.Begin : (j == (i + consumedTokens - 1) ? EntityTag.End : EntityTag.Inside))));
                            }
                        }

                        i += consumedTokens - 1; //-1 as we'll do an i++ imediatelly after
                        foundAny = true;
                        break;
                    }
                }
            }

            return foundAny;
        }

        public void NewPattern(string name, Action<MatchingPattern> pattern)
        {
            var mp = new MatchingPattern(name);
            pattern(mp);
            Data.Patterns.Add(mp);
        }

        public void RemovePattern(string name)
        {
            Data.Patterns.RemoveAll(p => p.Name == name);
        }
    }

    [MessagePackObject]
    public class MatchingPattern
    {
        [Key(0)] public List<PatternUnit[]> Patterns { get; set; } = new List<PatternUnit[]>();
        [Key(1)] public string Name { get; set; }

        public MatchingPattern()
        {
            //Constructor for Json serialization
        }

        [SerializationConstructor]
        public MatchingPattern(List<PatternUnit[]> patterns, string name)
        {
            Patterns = patterns;
            Name = name;
        }

        public MatchingPattern(IMatchingPattern prototype)
        {
            Patterns = ((MatchingPatternPrototype)prototype).Patterns.Select(p => p.Select(pt => new PatternUnit(pt)).ToArray()).ToList();
            Name = ((MatchingPatternPrototype)prototype).Name;
        }

        public MatchingPattern(string name)
        {
            Name = name;
        }

        public void From(MatchingPattern mp)
        {
            Patterns.AddRange(mp.Patterns);
        }

        public bool IsMatch(Span<Token> tokens, out int consumedTokens)
        {
            int largestMatch = -1;
            for (int i = 0; i < Patterns.Count; i++)
            {
                var currentToken = 0;
                for (int j = 0; j < Patterns[i].Length; j++)
                {
                    PatternUnit currentPattern = Patterns[i][j];
                    int ct = currentToken;
                    bool hasMatched = false;

                    while (ct < tokens.Length && currentPattern.IsMatch(ref tokens[ct]))
                    {
                        ct++;
                        hasMatched = true;
                        if (currentPattern.Mode == PatternMatchingMode.Single) { break; }
                    }

                    if (hasMatched)
                    {
                        currentToken = ct;
                    }
                    else
                    {
                        if (!currentPattern.Optional)
                        {
                            currentToken = int.MinValue; //Didn't match a mandatory token, so abort
                            break;
                        }
                    }
                }
                if (largestMatch < currentToken) { largestMatch = currentToken; }
            }

            if (largestMatch > 0)
            {
                consumedTokens = largestMatch;
                return true;
            }
            else
            {
                consumedTokens = 0;
                return false;
            }
        }

        public MatchingPattern Add(params IPatternUnit[] units)
        {
            Patterns.Add(units.Select(u => new PatternUnit(u)).ToArray());
            return this;
        }

        public MatchingPattern Add(params PatternUnit[] units)
        {
            Patterns.Add(units);
            return this;
        }
    }

    [MessagePackObject]
    public class PatternUnit
    {
        [Key(0)] public PatternMatchingMode Mode;
        [Key(1)] public bool Optional;
        [Key(2)] public bool CaseSensitive;
        [Key(3)] public PatternUnitType Type;
        [Key(4)] public PartOfSpeech[] POS;
        [Key(5)] public string Suffix;
        [Key(6)] public string Prefix;
        [Key(7)] public string Shape;
        [Key(8)] public string Token;
        [Key(9)] public HashSet<string> Set;
        [Key(10)] public string EntityType;
        [Key(11)] public HashSet<ulong> SetHashes;
        [Key(12)] public ulong TokenHash;
        [Key(13)] public PatternUnit LeftSide;
        [Key(14)] public PatternUnit RightSide;
        [Key(15)] public HashSet<char> ValidChars;
        [Key(16)] public int MinLength;
        [Key(17)] public int MaxLength;

        public PatternUnit(IPatternUnit prototype)
        {
            var p = (PatternUnitPrototype)prototype;
            Mode = p.Mode;
            Optional = p.Optional;
            CaseSensitive = p.CaseSensitive;
            Type = p.Type;
            POS = p.POS;
            Suffix = p.Suffix;
            Prefix = p.Prefix;
            Shape = p.Shape;
            Token = p.Token;
            Set = p.Set;
            EntityType = p.EntityType;
            SetHashes = p.SetHashes ?? (p.Set is null ? null : new HashSet<ulong>(p.Set.Select(token => p.CaseSensitive ? PatternUnitPrototype.Hash64(token.AsSpan()) : PatternUnitPrototype.IgnoreCaseHash64(token.AsSpan()))));
            TokenHash = p.TokenHash;
            LeftSide = p.LeftSide is object ? new PatternUnit(p.LeftSide) : null;
            RightSide = p.RightSide is object ? new PatternUnit(p.RightSide) : null;
            ValidChars = p.ValidChars;
            MinLength = p.MinLength;
            MaxLength = p.MaxLength;
        }

        public PatternUnit(Func<PatternUnitPrototype, PatternUnitPrototype> pattern) : this(pattern(new PatternUnitPrototype()))
        {
        }

        public PatternUnit()
        {
            //Constructor for Json serialization
        }

        [SerializationConstructor]
        public PatternUnit(PatternMatchingMode mode, bool optional, bool caseSensitive, PatternUnitType type, PartOfSpeech[] pos, string suffix, string prefix, string shape, string token, HashSet<string> set, string entityType, HashSet<ulong> setHashes, ulong tokenHash, PatternUnit leftSide, PatternUnit rightSide)
        {
            Mode = mode;
            Optional = optional;
            CaseSensitive = caseSensitive;
            Type = type;
            POS = pos;
            Suffix = suffix;
            Prefix = prefix;
            Shape = shape?.AsSpan().Shape(false);
            Token = token;
            Set = set;
            EntityType = entityType;
            SetHashes = setHashes ?? (set is null ? null : new HashSet<ulong>(set.Select(tk => CaseSensitive ? PatternUnitPrototype.Hash64(tk.AsSpan()) : PatternUnitPrototype.IgnoreCaseHash64(tk.AsSpan()))));
            TokenHash = tokenHash;
            LeftSide = leftSide;
            RightSide = rightSide;
        }

        #region Match

        public bool IsMatch(ref Token token)
        {
            bool isMatch = true;
            if (token.Length < 1) { return false; } //Empty tokens never match

            if (Mode == PatternMatchingMode.And)
            {
                return LeftSide.IsMatch(ref token) && RightSide.IsMatch(ref token);
            }
            else if (Mode == PatternMatchingMode.Or)
            {
                return LeftSide.IsMatch(ref token) || RightSide.IsMatch(ref token);
            }
            else
            {
                if (isMatch && (Type & PatternUnitType.Length) == PatternUnitType.Length) { isMatch &= MatchLength(ref token); }
                if (isMatch && (Type & PatternUnitType.Token) == PatternUnitType.Token) { isMatch &= MatchToken(ref token); }
                if (isMatch && (Type & PatternUnitType.Shape) == PatternUnitType.Shape) { isMatch &= MatchShape(ref token); }
                if (isMatch && (Type & PatternUnitType.WithChars) == PatternUnitType.WithChars) { isMatch &= MatchWithChars(ref token); }
                //if (isMatch && (Type & PatternUnitType.Script) == PatternUnitType.Script)                                 { isMatch &= MatchScript          (ref token); }
                if (isMatch && (Type & PatternUnitType.POS) == PatternUnitType.POS) { isMatch &= MatchPOS(ref token); }
                if (isMatch && (Type & PatternUnitType.MultiplePOS) == PatternUnitType.MultiplePOS) { isMatch &= MatchMultiplePOS(ref token); }
                if (isMatch && (Type & PatternUnitType.Suffix) == PatternUnitType.Suffix) { isMatch &= MatchSuffix(ref token); }
                if (isMatch && (Type & PatternUnitType.Prefix) == PatternUnitType.Prefix) { isMatch &= MatchPrefix(ref token); }
                if (isMatch && (Type & PatternUnitType.Set) == PatternUnitType.Set) { isMatch &= MatchSet(ref token); }
                if (isMatch && (Type & PatternUnitType.Entity) == PatternUnitType.Entity) { isMatch &= MatchEntity(ref token); }
                if (isMatch && (Type & PatternUnitType.NotEntity) == PatternUnitType.NotEntity) { isMatch &= !MatchEntity(ref token); }
                if (isMatch && (Type & PatternUnitType.IsDigit) == PatternUnitType.IsDigit) { isMatch &= MatchIsDigit(ref token); }
                if (isMatch && (Type & PatternUnitType.IsNumeric) == PatternUnitType.IsNumeric) { isMatch &= MatchIsNumeric(ref token); }
                if (isMatch && (Type & PatternUnitType.HasNumeric) == PatternUnitType.HasNumeric) { isMatch &= MatchHasNumeric(ref token); }
                if (isMatch && (Type & PatternUnitType.IsAlpha) == PatternUnitType.IsAlpha) { isMatch &= MatchIsAlpha(ref token); }
                if (isMatch && (Type & PatternUnitType.IsLetterOrDigit) == PatternUnitType.IsLetterOrDigit) { isMatch &= MatchIsLetterOrDigit(ref token); }
                //if (isMatch && (Type & PatternUnitType.IsLatin) == PatternUnitType.IsLatin)                               { isMatch &= MatchIsLatin         (ref token); }
                if (isMatch && (Type & PatternUnitType.IsEmoji) == PatternUnitType.IsEmoji) { isMatch &= MatchIsEmoji(ref token); }
                if (isMatch && (Type & PatternUnitType.IsPunctuation) == PatternUnitType.IsPunctuation) { isMatch &= MatchIsPunctuation(ref token); }
                if (isMatch && (Type & PatternUnitType.IsLowerCase) == PatternUnitType.IsLowerCase) { isMatch &= MatchIsLowerCase(ref token); }
                if (isMatch && (Type & PatternUnitType.IsUpperCase) == PatternUnitType.IsUpperCase) { isMatch &= MatchIsUpperCase(ref token); }
                if (isMatch && (Type & PatternUnitType.IsTitleCase) == PatternUnitType.IsTitleCase) { isMatch &= MatchIsTitleCase(ref token); }
                if (isMatch && (Type & PatternUnitType.LikeURL) == PatternUnitType.LikeURL) { isMatch &= MatchLikeURL(ref token); }
                if (isMatch && (Type & PatternUnitType.LikeEmail) == PatternUnitType.LikeEmail) { isMatch &= MatchLikeEmail(ref token); }
                if (isMatch && (Type & PatternUnitType.IsOpeningParenthesis) == PatternUnitType.IsOpeningParenthesis) { isMatch &= MatchIsOpeningParenthesis(ref token); }
                if (isMatch && (Type & PatternUnitType.IsClosingParenthesis) == PatternUnitType.IsClosingParenthesis) { isMatch &= MatchIsClosingParenthesis(ref token); }
            }

            return Mode == PatternMatchingMode.ShouldNotMatch ? !isMatch : isMatch;
        }

        private bool MatchLength(ref Token token)
        {
            bool isMatch = true;
            if (MinLength > 0)
            {
                isMatch &= token.Length >= MinLength;
            }
            if (MaxLength > 0)
            {
                isMatch &= token.Length <= MaxLength;
            }
            return isMatch;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ulong GetTokenHash(ref Token token)
        {
            return CaseSensitive ? PatternUnitPrototype.Hash64(token.ValueAsSpan) : PatternUnitPrototype.IgnoreCaseHash64(token.ValueAsSpan);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchToken(ref Token token)
        {
            return token.ValueAsSpan.Equals(Token.AsSpan(), CaseSensitive ? StringComparison.InvariantCulture : StringComparison.InvariantCultureIgnoreCase);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchWithChars(ref Token token)
        {
            foreach (var c in token.ValueAsSpan)
            {
                if (!ValidChars.Contains(c)) { return false; }
            }
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchShape(ref Token token)
        {
            return token.ValueAsSpan.Shape(false) == Shape;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchScript(ref Token token)
        {
            throw new NotImplementedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchPOS(ref Token token)
        {
            return token.POS == POS[0];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchMultiplePOS(ref Token token)
        {
            for (int i = 0; i < POS.Length; i++)
            {
                if (POS[i] == token.POS) { return true; }
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchSuffix(ref Token token)
        {
            return token.ValueAsSpan.EndsWith(Suffix.AsSpan(), CaseSensitive ? StringComparison.InvariantCulture : StringComparison.InvariantCultureIgnoreCase);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchPrefix(ref Token token)
        {
            return token.ValueAsSpan.StartsWith(Prefix.AsSpan(), CaseSensitive ? StringComparison.InvariantCulture : StringComparison.InvariantCultureIgnoreCase);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchSet(ref Token token)
        {
            if (SetHashes is null)
            {
                //No need to lock here, as we would just replace one with another equal set if there is a colision
                SetHashes = new HashSet<ulong>(Set.Select(tk => CaseSensitive ? PatternUnitPrototype.Hash64(tk.AsSpan()) : PatternUnitPrototype.IgnoreCaseHash64(tk.AsSpan())));
            }
            return SetHashes.Contains(GetTokenHash(ref token));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchEntity(ref Token token)
        {
            if (token.EntityTypes is null) { return false; }

            foreach (var et in token.EntityTypes)
            {
                if (et.Type == EntityType) { return true; }
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsDigit(ref Token token)
        {
            return token.ValueAsSpan.IsDigit();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsNumeric(ref Token token)
        {
            return token.ValueAsSpan.IsNumeric();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchHasNumeric(ref Token token)
        {
            return token.ValueAsSpan.HasNumeric();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsAlpha(ref Token token)
        {
            return token.ValueAsSpan.IsLetter();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsLetterOrDigit(ref Token token)
        {
            return token.ValueAsSpan.IsAllLetterOrDigit();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsLatin(ref Token token)
        {
            throw new NotImplementedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsEmoji(ref Token token)
        {
            return token.ValueAsSpan.IsEmoji(out var count) && count == token.Length; // Check if need this here - probably not as each emoji will be tokenized separately
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsPunctuation(ref Token token)
        {
            return token.ValueAsSpan.IsPunctuation();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsLowerCase(ref Token token)
        {
            return token.ValueAsSpan.IsAllLowerCase();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsUpperCase(ref Token token)
        {
            return token.ValueAsSpan.IsAllUpperCase();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsTitleCase(ref Token token)
        {
            return token.ValueAsSpan.IsCapitalized();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchLikeNumber(ref Token token)
        {
            throw new NotImplementedException();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchLikeURL(ref Token token)
        {
            return token.ValueAsSpan.IsLikeURLorEmail(); //TODO: Split these two in URL and Email
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchLikeEmail(ref Token token)
        {
            return token.ValueAsSpan.IsLikeURLorEmail();  //TODO: Split these two in URL and Email
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsOpeningParenthesis(ref Token token)
        {
            var span = token.ValueAsSpan;
            return span.Length == 1 && (span[0] == '(' || span[0] == '[');
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchIsClosingParenthesis(ref Token token)
        {
            var span = token.ValueAsSpan;
            return span.Length == 1 && (span[0] == ')' || span[0] == ']');
        }

        #endregion Match
    }

    //Used to serialize the pattern spotter model data for the front-end
    [MessagePackObject(keyAsPropertyName: true)]
    public class PatternSpotterData
    {
        public StoredObjectInfo Model { get; set; }
        public string CaptureTag { get; set; }
        public MatchingPattern[] Patterns { get; set; }
    }
}