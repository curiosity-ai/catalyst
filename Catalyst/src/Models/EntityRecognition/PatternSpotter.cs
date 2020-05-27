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

            var patterns = Data.Patterns; //copy local reference for the loop

            for (int i = 0; i < N; i++)
            {
                foreach (var p in patterns)
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
            //Creates new instances of all PatternUnits in the source MatchingPattern
            Patterns.AddRange(mp.Patterns.Select(pu => pu.Select(p => new PatternUnit(p.Mode, p.Optional, p.CaseSensitive, p.Type, p.POS, p.Suffix, p.Prefix, p.Shape, p.Token, p.Set, p.EntityType, p.LeftSide, p.RightSide)).ToArray()));
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
        [Key(0)] public PatternMatchingMode Mode { get; set; }
        [Key(1)] public bool Optional { get; set; }
        [Key(2)] public bool CaseSensitive { get => caseSensitive; set { caseSensitive = value; Set = set; } } //Reset Set so it recomputes the hashes based on the new case sensitivity
        [Key(3)] public PatternUnitType Type { get; set; }
        [Key(4)] public PartOfSpeech[] POS { get; set; }
        [Key(5)] public string Suffix { get => suffix; set { suffix = value; _splitSuffix = suffix?.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries)?.Distinct()?.ToArray(); } }
        [Key(6)] public string Prefix { get => prefix; set { prefix = value; _splitPrefix = prefix?.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries)?.Distinct()?.ToArray(); } }
        [Key(7)] public string Shape { get => shape; set { shape = value; _splitShape = !string.IsNullOrWhiteSpace(shape) ? new HashSet<string>(shape.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries).Select(s => s.AsSpan().Shape(compact: false))) : null; } }
        [Key(8)] public string Token { get; set; }
        [Key(9)] public string[] Set { get => set; set { set = value.Distinct().ToArray(); _setHashes = new HashSet<ulong>(set.Select(tk => CaseSensitive ? PatternUnitPrototype.Hash64(tk.AsSpan()) : PatternUnitPrototype.IgnoreCaseHash64(tk.AsSpan()))); } }
        [Key(10)] public string EntityType { get => entityType; set { entityType = value; _splitEntityType = entityType is object ? new HashSet<string>(entityType.Split(splitChar, StringSplitOptions.RemoveEmptyEntries)) : null; } }

        //[Key(11)] removed

        //[Key(12)] removed

        [Key(13)] public PatternUnit LeftSide { get; set; }
        [Key(14)] public PatternUnit RightSide { get; set; }
        [Key(15)] public HashSet<char> ValidChars { get; set; }
        [Key(16)] public int MinLength { get; set; }
        [Key(17)] public int MaxLength { get; set; }

        internal readonly static char[] splitChar = new[] { ',' };
        internal readonly static char[] splitCharWithWhitespaces = splitChar.Concat(CharacterClasses.WhitespaceCharacters).ToArray();

        private string[] _splitSuffix;
        private string[] _splitPrefix;
        private HashSet<string> _splitEntityType;
        private HashSet<string> _splitShape;
        private HashSet<ulong> _setHashes;

        private string suffix;
        private string prefix;
        private string shape;
        private string entityType;
        private string[] set;
        private bool caseSensitive;

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
            Set = p.Set.ToArray();
            EntityType = p.EntityType;
            LeftSide = p.LeftSide is object ? new PatternUnit(p.LeftSide) : null;
            RightSide = p.RightSide is object ? new PatternUnit(p.RightSide) : null;
            ValidChars = p.ValidChars;
            MinLength = p.MinLength;
            MaxLength = p.MaxLength;
        }

        //Constructor for Json/MsgPack serialization
        public PatternUnit()
        {
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
                //if (isMatch && (Type & PatternUnitType.Script) == PatternUnitType.Script)                                 { isMatch &= MatchScript(ref token); }
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
            return _splitShape is object && _splitShape.Contains(token.ValueAsSpan.Shape(compact: false));
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
            if (_splitSuffix is null) return false;

            foreach (var suffix in _splitSuffix)
            {
                if (token.ValueAsSpan.EndsWith(suffix.AsSpan(), CaseSensitive ? StringComparison.InvariantCulture : StringComparison.InvariantCultureIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchPrefix(ref Token token)
        {
            if (_splitPrefix is null) return false;

            foreach (var prefix in _splitPrefix)
            {
                if (token.ValueAsSpan.StartsWith(prefix.AsSpan(), CaseSensitive ? StringComparison.InvariantCulture : StringComparison.InvariantCultureIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchSet(ref Token token)
        {
            return _setHashes is object && _setHashes.Contains(GetTokenHash(ref token));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchEntity(ref Token token)
        {
            if (token.EntityTypes is null || _splitEntityType is null) { return false; }

            foreach (var et in token.EntityTypes)
            {
                if (_splitEntityType.Contains(et.Type)) { return true; }
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
            var span = token.ValueAsSpan;

            bool isLike = span.IsLikeURLorEmail();
            if (isLike)
            {
                if (span.IndexOf('@') > 0)
                {
                    if (span.IndexOf(':') > 0)
                    {
                        return true; //probably url with password
                    }
                    else
                    {
                        return false; //probably email
                    }
                }

                int countSlashDot = 0;
                int hasWWW = span.IndexOf(new[] { 'w', 'w', 'w' }) > 0 ? 5 : 0;
                int hasHTTP = span.IndexOf(new[] { 'h', 't', 't', 'p' }) > 0 ? 5 : 0;
                int hasFTP = span.IndexOf(new[] { 'f', 't', 'p' }) > 0 ? 5 : 0;

                for (int i = 0; i < span.Length; i++)
                {
                    if (span[i] == '.') countSlashDot++;
                    if (span[i] == ':') countSlashDot++;
                    if (span[i] == '/') countSlashDot++;
                }

                return countSlashDot + hasWWW + hasHTTP + hasFTP > 5;
            }
            else
            {
                return false;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchLikeEmail(ref Token token)
        {
            bool isLike = token.ValueAsSpan.IsLikeURLorEmail();
            if (isLike)
            {

                //TODO: refine these rules
                int countAt = 0;
                int countDot = 0;
                for (int i = 0; i < token.ValueAsSpan.Length; i++)
                {
                    if (token.ValueAsSpan[i] == '@') countAt++;
                    if (token.ValueAsSpan[i] == '.') countDot++;
                }
                return countAt == 1 && countDot > 1;
            }
            else
            {
                return false;
            }
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