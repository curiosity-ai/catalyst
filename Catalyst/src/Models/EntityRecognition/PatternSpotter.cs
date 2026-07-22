using MessagePack;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Catalyst.Algorithms;

namespace Catalyst.Models
{
    /// <summary>
    /// Represents the data model for the PatternSpotter.
    /// </summary>
    public class PatternSpotterModel : StorableObjectData
    {
        /// <summary>Gets or sets the tag to use for captured entities.</summary>
        public string CaptureTag { get; set; }
        /// <summary>Gets or sets the list of matching patterns.</summary>
        public List<MatchingPattern> Patterns { get; set; } = new List<MatchingPattern>();
    }

    /// <summary>
    /// Implements an entity recognizer that uses complex patterns of tokens to identify entities.
    /// </summary>
    public class PatternSpotter : StorableObjectV2<PatternSpotter, PatternSpotterModel>, IEntityRecognizer, IProcess
    {
        private PatternSpotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternSpotter"/> class.
        /// </summary>
        /// <param name="language">The language.</param>
        /// <param name="version">The version.</param>
        /// <param name="tag">The tag.</param>
        /// <param name="captureTag">The tag to use for captured entities.</param>
        public PatternSpotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        /// <summary>
        /// Loads a PatternSpotter from the store.
        /// </summary>
        /// <param name="language">The language.</param>
        /// <param name="version">The version.</param>
        /// <param name="tag">The tag.</param>
        /// <returns>A task that represents the asynchronous operation, returning the loaded spotter.</returns>
        public new static async Task<PatternSpotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new PatternSpotter(language, version, tag);
            await a.LoadDataAsync();
            return a;
        }

        /// <inheritdoc />
        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            RecognizeEntities(document);
        }

        /// <inheritdoc />
        public string[] Produces()
        {
            return new[] { Data.CaptureTag };
        }

        /// <summary>
        /// Checks if the document contains any entities recognized by this spotter.
        /// </summary>
        /// <param name="document">The document.</param>
        /// <returns>True if any entity is found.</returns>
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

        /// <inheritdoc />
        public bool RecognizeEntities(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span, stopOnFirstFound: false);
            }
            return foundAny;
        }

        /// <summary>
        /// Clears all patterns from the model.
        /// </summary>
        public void ClearModel()
        {
            Data.Patterns.Clear();
        }

        /// <summary>
        /// Recognizes entities in the specified span.
        /// </summary>
        /// <param name="ispan">The span.</param>
        /// <param name="stopOnFirstFound">Whether to stop after finding the first entity.</param>
        /// <returns>True if any entities were found.</returns>
        public bool RecognizeEntities(Span ispan, bool stopOnFirstFound = false)
        {
            var pooledTokens = ispan.ToTokenSpanPolled(out var actualLength);
            var tokens = pooledTokens.AsSpan(0, actualLength);

            int N = tokens.Length;
            bool foundAny = false;

            var patterns = Data.Patterns; //copy local reference for the loop

            for (int i = 0; i < N; i++)
            {
                int consumedTokens = -1;

                foreach (var p in patterns) //Loop through all patterns as we want to find the longest match possible
                {
                    if (p.IsMatch(tokens.Slice(i), out var consumedTokensByThisPattern))
                    {
                        if (stopOnFirstFound) { return true; } //Do not capture the entity, just returns true to say there's a match

                        consumedTokens = Math.Max(consumedTokens, consumedTokensByThisPattern);
                    }
                }

                if (consumedTokens > 0)
                {
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
                }
            }

            ArrayPool<Token>.Shared.Return(pooledTokens);

            return foundAny;
        }

        /// <summary>
        /// Returns a per-position trace of all pattern match attempts. Read-only;
        /// this method does not mutate the document. Significantly more allocating
        /// than <see cref="Process"/>; intended for diagnostics only.
        /// Unlike <see cref="RecognizeEntities(IDocument)"/>, this reports every
        /// starting position the matcher would consider, including positions inside
        /// an already-matched span, so callers can answer "why didn't a match start here?".
        /// </summary>
        /// <param name="document">The document to analyze.</param>
        /// <param name="cancellationToken">A cancellation token.</param>
        /// <returns>The list of per-attempt explanations.</returns>
        public IReadOnlyList<PatternMatchExplanation> Explain(IDocument document, CancellationToken cancellationToken = default)
        {
            var result = new List<PatternMatchExplanation>();
            var patterns = Data.Patterns;
            int spanIndex = 0;

            foreach (var span in document)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var pooledTokens = span.ToTokenSpanPolled(out var actualLength);
                try
                {
                    var tokens = pooledTokens.AsSpan(0, actualLength);
                    int N = tokens.Length;

                    for (int i = 0; i < N; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        foreach (var p in patterns)
                        {
                            var trace = new MatchTraceBuilder();
                            p.IsMatch(tokens.Slice(i), i, trace, out var consumed);
                            result.Add(new PatternMatchExplanation
                            {
                                PatternName = p.Name,
                                SpanIndex = spanIndex,
                                StartTokenIndex = i,
                                ConsumedTokens = consumed,
                                Alternatives = trace.Drain(consumed)
                            });
                        }
                    }
                }
                finally
                {
                    ArrayPool<Token>.Shared.Return(pooledTokens);
                }
                spanIndex++;
            }

            return result;
        }

        /// <summary>
        /// Adds a new pattern to the spotter.
        /// </summary>
        /// <param name="name">The name of the pattern.</param>
        /// <param name="pattern">The action to configure the pattern.</param>
        public void NewPattern(string name, Action<MatchingPattern> pattern)
        {
            var mp = new MatchingPattern(name);
            pattern(mp);
            Data.Patterns.Add(mp);
        }

        /// <summary>
        /// Removes a pattern by name.
        /// </summary>
        /// <param name="name">The name of the pattern to remove.</param>
        public void RemovePattern(string name)
        {
            Data.Patterns.RemoveAll(p => p.Name == name);
        }
    }

    /// <summary>
    /// Represents a pattern composed of multiple sequences of pattern units.
    /// </summary>
    [MessagePackObject]
    public class MatchingPattern
    {
        /// <summary>Gets or sets the list of alternative sequences of pattern units.</summary>
        [Key(0)] public List<PatternUnit[]> Patterns { get; set; } = new List<PatternUnit[]>();
        /// <summary>Gets or sets the name of the pattern.</summary>
        [Key(1)] public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchingPattern"/> class.
        /// </summary>
        public MatchingPattern()
        {
            //Constructor for Json serialization
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchingPattern"/> class with specified patterns and name.
        /// </summary>
        /// <param name="patterns">The patterns.</param>
        /// <param name="name">The name.</param>
        [SerializationConstructor]
        public MatchingPattern(List<PatternUnit[]> patterns, string name)
        {
            Patterns = patterns;
            Name = name;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchingPattern"/> class from a prototype.
        /// </summary>
        /// <param name="prototype">The prototype.</param>
        public MatchingPattern(IMatchingPattern prototype)
        {
            Patterns = ((MatchingPatternPrototype)prototype).Patterns.Select(p => p.Select(pt => new PatternUnit(pt)).ToArray()).ToList();
            Name = ((MatchingPatternPrototype)prototype).Name;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchingPattern"/> class with a specified name.
        /// </summary>
        /// <param name="name">The name.</param>
        public MatchingPattern(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Copies patterns from another <see cref="MatchingPattern"/>.
        /// </summary>
        /// <param name="mp">The source pattern.</param>
        public void From(MatchingPattern mp)
        {
            //Creates new instances of all PatternUnits in the source MatchingPattern
            Patterns.AddRange(mp.Patterns.Select(pu => pu.Select(p => p.Clone()).ToArray()));
        }

        /// <summary>
        /// Checks if the specified tokens match any of the sequences in this pattern.
        /// </summary>
        /// <param name="tokens">The tokens to match.</param>
        /// <param name="consumedTokens">The number of tokens consumed by the match.</param>
        /// <returns>True if it's a match.</returns>
        public bool IsMatch(Span<Token> tokens, out int consumedTokens)
        {
            int largestMatch = -1;
            var patterns = Patterns;

            for (int i = 0; i < patterns.Count; i++)
            {
                var currentToken = 0;
                var innerPattern = patterns[i];
                for (int j = 0; j < innerPattern.Length; j++)
                {
                    var currentPattern = innerPattern[j];
                    int ct = currentToken;
                    
                    int maxMatches = currentPattern.MaxMatches;
                    
                    if (maxMatches == 0) maxMatches = 10;

                    bool hasMatched = false;

                    while (ct < tokens.Length && currentPattern.IsMatch(ref tokens[ct]) && maxMatches > 0)
                    {
                        ct++;
                        hasMatched = true;
                        if (currentPattern.Mode == PatternMatchingMode.Single) 
                        {
                            break;
                        }
                        else
                        {
                            maxMatches--; //Limits the number of multiple matches
                        }
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

        // Traced sibling of IsMatch. Mirrors the production matcher's largest-match
        // selection but records a UnitTrace per while-loop iteration and an
        // AlternativeTrace per alternative into the supplied builder.
        internal bool IsMatch(Span<Token> tokens, int absoluteStart, MatchTraceBuilder trace, out int consumedTokens)
        {
            int largestMatch = -1;
            var patterns = Patterns;

            for (int i = 0; i < patterns.Count; i++)
            {
                trace.BeginAlternative(i);
                var currentToken = 0;
                var innerPattern = patterns[i];
                for (int j = 0; j < innerPattern.Length; j++)
                {
                    var currentPattern = innerPattern[j];
                    int ct = currentToken;

                    int maxMatches = currentPattern.MaxMatches;
                    if (maxMatches == 0) maxMatches = 10;

                    bool hasMatched = false;
                    bool didAttempt = false;

                    while (ct < tokens.Length && maxMatches > 0)
                    {
                        string tokenVal = tokens[ct].ValueAsSpan.ToString();
                        trace.BeginUnit(j, currentPattern.Mode, currentPattern.Optional, absoluteStart + ct, tokenVal);
                        bool unitMatched = currentPattern.IsMatch(ref tokens[ct], trace);
                        var ut = trace.EndUnit(unitMatched);
                        trace.AddUnitToAlternative(ut);
                        didAttempt = true;

                        if (!unitMatched) break;

                        ct++;
                        hasMatched = true;
                        if (currentPattern.Mode == PatternMatchingMode.Single) break;
                        maxMatches--;
                    }

                    // If the loop never entered (ct >= tokens.Length), still record a synthetic
                    // failure trace so the explanation reflects that we couldn't even attempt.
                    if (!didAttempt)
                    {
                        trace.BeginUnit(j, currentPattern.Mode, currentPattern.Optional, absoluteStart + ct, string.Empty);
                        trace.RecordCriterion(MatchCriterion.TokenLengthZero, false, null, null);
                        var ut = trace.EndUnit(false);
                        trace.AddUnitToAlternative(ut);
                    }

                    if (hasMatched)
                    {
                        currentToken = ct;
                    }
                    else
                    {
                        if (!currentPattern.Optional)
                        {
                            currentToken = int.MinValue;
                            break;
                        }
                    }
                }

                bool altCompleted = currentToken > 0;
                int altConsumed = altCompleted ? currentToken : 0;
                trace.EndAlternative(altCompleted, altConsumed);

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

        /// <summary>
        /// Adds a sequence of pattern units to this pattern.
        /// </summary>
        /// <param name="units">The pattern units.</param>
        /// <returns>This instance.</returns>
        public MatchingPattern Add(params IPatternUnit[] units)
        {
            Patterns.Add(units.Select(u => new PatternUnit(u)).ToArray());
            return this;
        }

        /// <summary>
        /// Adds a sequence of pattern units to this pattern.
        /// </summary>
        /// <param name="units">The pattern units.</param>
        /// <returns>This instance.</returns>
        public MatchingPattern Add(params PatternUnit[] units)
        {
            Patterns.Add(units);
            return this;
        }
    }

    public class ShapeEqualityComparer : IEqualityComparer<string>
    {
        public static readonly ShapeEqualityComparer Instance = new ShapeEqualityComparer();

        public bool Equals(string x, string y)
        {
            if (x == null && y == null) return true;
            if (x == null || y == null) return false;
            if (x.Length != y.Length) return false;

            for (int i = 0; i < x.Length; i++)
            {
                if (x[i] == y[i]) continue;
                if (x[i] == '~' && IsValidShapeChar(y[i])) continue;
                if (y[i] == '~' && IsValidShapeChar(x[i])) continue;
                return false;
            }
            return true;
        }

        private bool IsValidShapeChar(char c)
        {
            return c == 'x' || c == 'X' || c == '9';
        }

        public int GetHashCode(string obj)
        {
            if (obj == null) return 0;
            int hash = 17;
            foreach (char c in obj)
            {
                char h = c;
                //We need to normalize the valid alpha-numeric shape characters to the same value for the hash code
                if (h == 'x' || h == 'X' || h == '9' || h == '~') h = '~';
                hash = hash * 31 + h.GetHashCode();
            }
            return hash;
        }
    }

    /// <summary>
    /// Represents a single unit in a pattern, which can match a token based on various criteria.
    /// </summary>
    [MessagePackObject]
    public class PatternUnit
    {
        /// <summary>Gets or sets the matching mode (Single, And, Or, ShouldNotMatch).</summary>
        [Key(0)] public PatternMatchingMode Mode { get; set; } = PatternMatchingMode.Single;
        /// <summary>Gets or sets a value indicating whether this unit is optional.</summary>
        [Key(1)] public bool Optional { get; set; }
        /// <summary>Gets or sets a value indicating whether matching should be case-sensitive.</summary>
        [Key(2)] public bool CaseSensitive { get => caseSensitive; set { caseSensitive = value; Set = set; } } //Reset Set so it recomputes the hashes based on the new case sensitivity
        /// <summary>Gets or sets the criteria types used for matching.</summary>
        [Key(3)] public PatternUnitType Type { get; set; }
        /// <summary>Gets or sets the part of speech to match.</summary>
        [Key(4)] public PartOfSpeech[] POS { get; set; }
        /// <summary>Gets or sets the suffix to match.</summary>
        [Key(5)] public string Suffix { get => suffix; set { suffix = value; _splitSuffix = suffix?.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries)?.Distinct()?.ToArray(); } }
        /// <summary>Gets or sets the prefix to match.</summary>
        [Key(6)] public string Prefix { get => prefix; set { prefix = value; _splitPrefix = prefix?.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries)?.Distinct()?.ToArray(); } }
        /// <summary>Gets or sets the shape to match.</summary>
        [Key(7)] public string Shape { get => shape; set { shape = value; _splitShape = !string.IsNullOrWhiteSpace(shape) ? new HashSet<string>(shape.Split(splitCharWithWhitespaces, StringSplitOptions.RemoveEmptyEntries).Select(s => s.AsSpan().Shape(compact: false)), ShapeEqualityComparer.Instance) : null; } }
        /// <summary>Gets or sets the exact token value to match.</summary>
        [Key(8)] public string Token { get; set; }
        /// <summary>Gets or sets the set of token values to match.</summary>
        [Key(9)] public string[] Set { get => set; set { set = value?.Distinct()?.ToArray(); RebuildSetHashes(); } }
        /// <summary>Gets or sets the entity type to match.</summary>
        [Key(10)] public string EntityType { get => entityType; set { entityType = value; _splitEntityType = entityType is object ? new HashSet<string>(entityType.Split(splitChar, StringSplitOptions.RemoveEmptyEntries)) : null; } }

        //[Key(11)] removed

        //[Key(12)] removed

        /// <summary>Gets or sets the left side of an And/Or/Not operation.</summary>
        [Key(13)] public PatternUnit LeftSide { get; set; }
        /// <summary>Gets or sets the right side of an And/Or operation.</summary>
        [Key(14)] public PatternUnit RightSide { get; set; }
        /// <summary>Gets or sets the set of valid characters.</summary>
        [Key(15)] public HashSet<char> ValidChars { get; set; }
        /// <summary>Gets or sets the minimum length of the token value.</summary>
        [Key(16)] public int MinLength { get; set; }
        /// <summary>Gets or sets the maximum length of the token value.</summary>
        [Key(17)] public int MaxLength { get; set; }
        /// <summary>Gets or sets the maximum number of times this unit can match consecutively.</summary>
        [Key(18)] public int MaxMatches { get; set; }
        [Key(19)] public float Confidence { get; set; }

        internal readonly static char[] splitChar = new[] { ',' };
        internal readonly static char[] splitCharWithWhitespaces = splitChar.Concat(CharacterClasses.WhitespaceCharacters).ToArray();

        private string[] _splitSuffix;
        private string[] _splitPrefix;
        private HashSet<string> _splitEntityType;
        private HashSet<string> _splitShape;
        private HashSet<ulong> _setHashes;
        private int _setMinLength;
        private int _setMaxLength;

        private string suffix;
        private string prefix;
        private string shape;
        private string entityType;
        private string[] set;
        private bool caseSensitive;
        private PatternUnit p;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternUnit"/> class from a prototype.
        /// </summary>
        /// <param name="prototype">The prototype.</param>
        public PatternUnit(IPatternUnit prototype)
        {
            var p = (PatternUnitPrototype)prototype;
            Mode          = p.Mode;
            Optional      = p.Optional;
            CaseSensitive = p.CaseSensitive;
            Type          = p.Type;
            POS           = p.POS;
            Suffix        = p.Suffix;
            Prefix        = p.Prefix;
            Shape         = p.Shape;
            Token         = p.Token;
            Set           = p.Set?.ToArray() ?? Array.Empty<string>();
            EntityType    = p.EntityType;
            LeftSide      = p.LeftSide is object ? new PatternUnit(p.LeftSide) : null;
            RightSide     = p.RightSide is object ? new PatternUnit(p.RightSide) : null;
            ValidChars    = p.ValidChars;
            MinLength     = p.MinLength;
            MaxLength     = p.MaxLength;
            MaxMatches    = p.MaxMatches;
            Confidence    = p.Confidence;
        }

        //Constructor for Json/MsgPack serialization
        /// <summary>
        /// Initializes a new instance of the <see cref="PatternUnit"/> class.
        /// </summary>
        public PatternUnit()
        {
        }

        /// <summary>
        /// Clones this instance.
        /// </summary>
        /// <returns>A new <see cref="PatternUnit"/> instance.</returns>
        public PatternUnit Clone()
        {
            return new PatternUnit()
            {
                Mode          = Mode,
                Optional      = Optional,
                CaseSensitive = CaseSensitive,
                Type          = Type,
                POS           = POS,
                Suffix        = Suffix,
                Prefix        = Prefix,
                Shape         = Shape,
                Token         = Token,
                Set           = Set,
                EntityType    = EntityType,
                LeftSide      = LeftSide,
                RightSide     = RightSide,
                ValidChars    = ValidChars,
                MinLength     = MinLength,
                MaxLength     = MaxLength,
                MaxMatches    = MaxMatches,
                Confidence    = Confidence
            };
        }

        #region Match

        /// <summary>
        /// Checks if the specified token matches the criteria in this unit.
        /// </summary>
        /// <param name="token">The token to match.</param>
        /// <returns>True if it matches.</returns>
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
                if (isMatch && (Type & PatternUnitType.Fuzzy) == PatternUnitType.Fuzzy) { isMatch &= MatchTokenFuzzy(ref token); }
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
        private bool MatchTokenFuzzy(ref Token token) 
        {
            ReadOnlySpan<char> token1;
            ReadOnlySpan<char> token2;
            if (CaseSensitive) 
            {
                token1 = token.ValueAsSpan;
                token2 = Token.AsSpan();
            } 
            else 
            {
                token1 = token.Value.ToLowerInvariant().AsSpan();
                token2 = Token.ToLowerInvariant().AsSpan();
            }
            int distance = Levenshtein.GetDistance( token1, token2, true);
            return 1.0-((float)distance / Math.Max(token1.Length, token2.Length)) >= Confidence;
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

        // Precomputes the set-membership hashes together with the [min, max] character-length window of the set
        // entries. The length window is the source for the pre-filter in MatchSet - a token can only equal one of
        // the set entries if its length matches one of them, so we can reject out-of-range tokens without hashing.
        private void RebuildSetHashes()
        {
            if (set is null)
            {
                _setHashes    = null;
                _setMinLength = 0;
                _setMaxLength = 0;
                return;
            }

            _setHashes = new HashSet<ulong>(set.Select(tk => CaseSensitive ? PatternUnitPrototype.Hash64(tk.AsSpan()) : PatternUnitPrototype.IgnoreCaseHash64(tk.AsSpan())));

            int min = int.MaxValue, max = 0;
            foreach (var tk in set)
            {
                int len = tk?.Length ?? 0;
                if (len < min) { min = len; }
                if (len > max) { max = len; }
            }
            _setMinLength = min;
            _setMaxLength = max;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MatchSet(ref Token token)
        {
            if (_setHashes is null) { return false; }
            int length = token.Length;
            if (length < _setMinLength || length > _setMaxLength) { return false; } //Length pre-filter: a token outside the set's length window cannot be in the set, so skip hashing
            return _setHashes.Contains(GetTokenHash(ref token));
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
                return countAt == 1 && countDot >= 1;
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

        #region Match (traced)

        // Traced sibling of IsMatch. Records every criterion evaluation (pass or fail)
        // into the supplied MatchTraceBuilder. Unlike the hot-path overload this
        // does NOT short-circuit on the first failing criterion: it continues so the
        // trace reveals every failing criterion rather than just the first one.
        internal bool IsMatch(ref Token token, MatchTraceBuilder trace)
        {
            if (token.Length < 1)
            {
                trace.RecordCriterion(MatchCriterion.TokenLengthZero, false, null, "");
                return false;
            }

            if (Mode == PatternMatchingMode.And)
            {
                var left = CaptureChild(LeftSide, ref token, trace);
                var right = CaptureChild(RightSide, ref token, trace);
                trace.SetLeftRight(left, right);
                bool result = left.Matched && right.Matched;
                trace.RecordCriterion(MatchCriterion.And, result, null, null);
                return result;
            }
            else if (Mode == PatternMatchingMode.Or)
            {
                var left = CaptureChild(LeftSide, ref token, trace);
                var right = CaptureChild(RightSide, ref token, trace);
                trace.SetLeftRight(left, right);
                bool result = left.Matched || right.Matched;
                trace.RecordCriterion(MatchCriterion.Or, result, null, null);
                return result;
            }

            bool isMatch = true;
            string tokenValue = null; // lazily materialized below if needed

            if ((Type & PatternUnitType.Length) == PatternUnitType.Length)
            {
                bool passed = MatchLength(ref token);
                trace.RecordCriterion(MatchCriterion.Length, passed, $"{MinLength}..{MaxLength}", token.Length.ToString());
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Token) == PatternUnitType.Token)
            {
                bool passed = MatchToken(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.Token, passed, Token, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Shape) == PatternUnitType.Shape)
            {
                bool passed = MatchShape(ref token);
                trace.RecordCriterion(MatchCriterion.Shape, passed, Shape, null);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.WithChars) == PatternUnitType.WithChars)
            {
                bool passed = MatchWithChars(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.WithChars, passed, ValidChars is null ? null : new string(System.Linq.Enumerable.ToArray(ValidChars)), tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.POS) == PatternUnitType.POS)
            {
                bool passed = MatchPOS(ref token);
                trace.RecordCriterion(MatchCriterion.POS, passed, POS != null && POS.Length > 0 ? POS[0].ToString() : null, token.POS.ToString());
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.MultiplePOS) == PatternUnitType.MultiplePOS)
            {
                bool passed = MatchMultiplePOS(ref token);
                trace.RecordCriterion(MatchCriterion.MultiplePOS, passed, POS == null ? null : string.Join("|", POS), token.POS.ToString());
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Suffix) == PatternUnitType.Suffix)
            {
                bool passed = MatchSuffix(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.Suffix, passed, Suffix, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Prefix) == PatternUnitType.Prefix)
            {
                bool passed = MatchPrefix(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.Prefix, passed, Prefix, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Set) == PatternUnitType.Set)
            {
                bool passed = MatchSet(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.Set, passed, Set == null ? null : string.Join("|", Set), tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.Entity) == PatternUnitType.Entity)
            {
                bool passed = MatchEntity(ref token);
                trace.RecordCriterion(MatchCriterion.Entity, passed, EntityType, FormatEntityTypes(ref token));
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.NotEntity) == PatternUnitType.NotEntity)
            {
                bool passed = !MatchEntity(ref token);
                trace.RecordCriterion(MatchCriterion.NotEntity, passed, EntityType, FormatEntityTypes(ref token));
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsDigit) == PatternUnitType.IsDigit)
            {
                bool passed = MatchIsDigit(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsDigit, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsNumeric) == PatternUnitType.IsNumeric)
            {
                bool passed = MatchIsNumeric(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsNumeric, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.HasNumeric) == PatternUnitType.HasNumeric)
            {
                bool passed = MatchHasNumeric(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.HasNumeric, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsAlpha) == PatternUnitType.IsAlpha)
            {
                bool passed = MatchIsAlpha(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsAlpha, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsLetterOrDigit) == PatternUnitType.IsLetterOrDigit)
            {
                bool passed = MatchIsLetterOrDigit(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsLetterOrDigit, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsEmoji) == PatternUnitType.IsEmoji)
            {
                bool passed = MatchIsEmoji(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsEmoji, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsPunctuation) == PatternUnitType.IsPunctuation)
            {
                bool passed = MatchIsPunctuation(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsPunctuation, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsLowerCase) == PatternUnitType.IsLowerCase)
            {
                bool passed = MatchIsLowerCase(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsLowerCase, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsUpperCase) == PatternUnitType.IsUpperCase)
            {
                bool passed = MatchIsUpperCase(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsUpperCase, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsTitleCase) == PatternUnitType.IsTitleCase)
            {
                bool passed = MatchIsTitleCase(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsTitleCase, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.LikeURL) == PatternUnitType.LikeURL)
            {
                bool passed = MatchLikeURL(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.LikeURL, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.LikeEmail) == PatternUnitType.LikeEmail)
            {
                bool passed = MatchLikeEmail(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.LikeEmail, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsOpeningParenthesis) == PatternUnitType.IsOpeningParenthesis)
            {
                bool passed = MatchIsOpeningParenthesis(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsOpeningParenthesis, passed, null, tokenValue);
                isMatch &= passed;
            }
            if ((Type & PatternUnitType.IsClosingParenthesis) == PatternUnitType.IsClosingParenthesis)
            {
                bool passed = MatchIsClosingParenthesis(ref token);
                tokenValue ??= token.ValueAsSpan.ToString();
                trace.RecordCriterion(MatchCriterion.IsClosingParenthesis, passed, null, tokenValue);
                isMatch &= passed;
            }

            if (Mode == PatternMatchingMode.ShouldNotMatch)
            {
                bool final = !isMatch;
                trace.RecordCriterion(MatchCriterion.ShouldNotMatch, final, null, null);
                return final;
            }
            return isMatch;
        }

        private static string FormatEntityTypes(ref Token token)
        {
            var ets = token.EntityTypes;
            if (ets is null || ets.Count == 0) return null;
            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < ets.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(ets[i].Type);
            }
            return sb.ToString();
        }

        private static UnitTrace CaptureChild(PatternUnit child, ref Token token, MatchTraceBuilder trace)
        {
            int tokIdx = trace.CurrentUnitTokenIndex;
            string tokVal = trace.CurrentUnitTokenValue;
            var saved = trace.SaveUnit();
            trace.BeginUnit(-1, child.Mode, child.Optional, tokIdx, tokVal);
            bool matched = child.IsMatch(ref token, trace);
            var ut = trace.EndUnit(matched);
            trace.RestoreUnit(saved);
            return ut;
        }

        #endregion Match (traced)
    }

    //Used to serialize the pattern spotter model data for the front-end
    /// <summary>
    /// Represents the serialized data for the PatternSpotter, intended for frontend use.
    /// </summary>
    [MessagePackObject(keyAsPropertyName: true)]
    public class PatternSpotterData
    {
        /// <summary>Gets or sets the model information.</summary>
        public StoredObjectInfo Model { get; set; }
        /// <summary>Gets or sets the capture tag.</summary>
        public string CaptureTag { get; set; }
        /// <summary>Gets or sets the matching patterns.</summary>
        public MatchingPattern[] Patterns { get; set; }
    }
}