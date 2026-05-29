using System.Collections.Generic;

#if !NET5_0_OR_GREATER
namespace System.Runtime.CompilerServices
{
    // Polyfill so `init`-only setters can be used when targeting netstandard2.1 / netcoreapp3.1.
    internal static class IsExternalInit { }
}
#endif

namespace Catalyst.Models
{
    /// <summary>
    /// Identifies which class of matching criterion in a <see cref="PatternUnit"/> was evaluated.
    /// Used by <see cref="PatternSpotter.Explain(IDocument, System.Threading.CancellationToken)"/>.
    /// </summary>
    public enum MatchCriterion
    {
        Length, Token, Shape, WithChars, POS, MultiplePOS,
        Suffix, Prefix, Set, Entity, NotEntity,
        IsDigit, IsNumeric, HasNumeric, IsAlpha, IsLetterOrDigit,
        IsEmoji, IsPunctuation, IsLowerCase, IsUpperCase, IsTitleCase,
        LikeURL, LikeEmail, IsOpeningParenthesis, IsClosingParenthesis,
        And, Or, ShouldNotMatch, TokenLengthZero
    }

    /// <summary>
    /// A single criterion evaluation inside a <see cref="UnitTrace"/>.
    /// </summary>
    public sealed class CriterionTrace
    {
        /// <summary>Which criterion class was evaluated.</summary>
        public MatchCriterion Criterion { get; init; }
        /// <summary>Whether this criterion passed.</summary>
        public bool Passed { get; init; }
        /// <summary>The configured expected value (e.g. token literal, "NOUN|VERB"). Null when not applicable.</summary>
        public string Expected { get; init; }
        /// <summary>The actual value observed on the token. Null when not applicable.</summary>
        public string Actual { get; init; }
    }

    /// <summary>
    /// The trace of a single <see cref="PatternUnit"/> evaluation against a specific token position.
    /// </summary>
    public sealed class UnitTrace
    {
        /// <summary>The index of the unit within its alternative. -1 when this trace is a child of an And/Or unit.</summary>
        public int UnitIndex { get; init; }
        /// <summary>The matching mode of the source unit.</summary>
        public PatternMatchingMode Mode { get; init; }
        /// <summary>Whether the source unit was optional.</summary>
        public bool Optional { get; init; }
        /// <summary>The absolute index of the evaluated token in the span.</summary>
        public int TokenIndex { get; init; }
        /// <summary>The evaluated token's value.</summary>
        public string TokenValue { get; init; }
        /// <summary>The final result, after ShouldNotMatch inversion and And/Or combination.</summary>
        public bool Matched { get; init; }
        /// <summary>The individual criteria evaluated for this unit (empty for And/Or units).</summary>
        public List<CriterionTrace> Criteria { get; init; } = new();
        /// <summary>For And/Or units: the trace of the left sub-unit.</summary>
        public UnitTrace LeftSide { get; init; }
        /// <summary>For And/Or units: the trace of the right sub-unit.</summary>
        public UnitTrace RightSide { get; init; }
    }

    /// <summary>
    /// The outcome of evaluating one alternative within a <see cref="MatchingPattern"/>.
    /// Distinguishes "completed but a longer alternative beat it" from "didn't complete at all",
    /// so a human reader can tell why a given alternative is or isn't the explanation
    /// for the pattern's consumed-token count.
    /// </summary>
    public enum AlternativeOutcome
    {
        /// <summary>A mandatory unit didn't match; this alternative never completed.</summary>
        Failed,
        /// <summary>The alternative completed but another alternative consumed more tokens and won.</summary>
        LostShorter,
        /// <summary>The alternative's consumed-token count is the one the pattern reports. Ties win.</summary>
        Won,
    }

    /// <summary>
    /// The trace of one alternative inside a <see cref="MatchingPattern"/>.
    /// </summary>
    public sealed class AlternativeTrace
    {
        /// <summary>The index of this alternative in the pattern's alternatives list.</summary>
        public int AlternativeIndex { get; init; }
        /// <summary>The unit traces in evaluation order. Multiple-mode units may produce multiple entries.</summary>
        public List<UnitTrace> Units { get; init; } = new();
        /// <summary>How this alternative fared against the rest of the pattern's alternatives.</summary>
        public AlternativeOutcome Outcome { get; init; }
        /// <summary>The number of tokens this alternative consumed (0 when <see cref="Outcome"/> is <see cref="AlternativeOutcome.Failed"/>).</summary>
        public int ConsumedTokens { get; init; }
    }

    /// <summary>
    /// The full explanation of one pattern's match attempt starting at a specific token index.
    /// </summary>
    public sealed class PatternMatchExplanation
    {
        /// <summary>The name of the pattern that was tried.</summary>
        public string PatternName { get; init; }
        /// <summary>The zero-based index of the span (sentence) within the document.</summary>
        public int SpanIndex { get; init; }
        /// <summary>The token index within its span where the matcher started.</summary>
        public int StartTokenIndex { get; init; }
        /// <summary>The number of tokens consumed by the winning alternative, or 0 if none matched.</summary>
        public int ConsumedTokens { get; init; }
        /// <summary>The trace of each alternative tried by the pattern.</summary>
        public List<AlternativeTrace> Alternatives { get; init; } = new();
    }

    /// <summary>
    /// Mutable scratchpad used while building per-attempt traces. Passed by reference through the matcher.
    /// </summary>
    internal sealed class MatchTraceBuilder
    {
        // Rows are kept in their pre-finalization shape (completed + consumed) because
        // the winner is only known after every alternative has been evaluated.
        private sealed class PendingAlternative
        {
            public int Index;
            public List<UnitTrace> Units;
            public bool Completed;
            public int Consumed;
        }

        private readonly List<PendingAlternative> _alternatives = new();

        // pending alternative
        private int _altIndex;
        private List<UnitTrace> _altUnits;

        // pending unit scratch
        private int _unitIndex;
        private PatternMatchingMode _unitMode;
        private bool _unitOptional;
        private int _unitTokenIndex;
        private string _unitTokenValue;
        private List<CriterionTrace> _unitCriteria;
        private UnitTrace _unitLeft;
        private UnitTrace _unitRight;

        public void BeginAlternative(int index)
        {
            _altIndex = index;
            _altUnits = new List<UnitTrace>();
        }

        // `completed` reflects whether the alternative ran its full unit sequence
        // without aborting on a mandatory unit; whether it Won or LostShorter is
        // decided later in Drain once the winning consumed-count is known.
        public void EndAlternative(bool completed, int consumed)
        {
            _alternatives.Add(new PendingAlternative
            {
                Index = _altIndex,
                Units = _altUnits ?? new List<UnitTrace>(),
                Completed = completed,
                Consumed = consumed,
            });
            _altUnits = null;
        }

        public void BeginUnit(int unitIndex, PatternMatchingMode mode, bool optional, int tokenIndex, string tokenValue)
        {
            _unitIndex = unitIndex;
            _unitMode = mode;
            _unitOptional = optional;
            _unitTokenIndex = tokenIndex;
            _unitTokenValue = tokenValue;
            _unitCriteria = new List<CriterionTrace>();
            _unitLeft = null;
            _unitRight = null;
        }

        public void RecordCriterion(MatchCriterion crit, bool passed, string expected, string actual)
        {
            _unitCriteria.Add(new CriterionTrace
            {
                Criterion = crit,
                Passed = passed,
                Expected = expected,
                Actual = actual
            });
        }

        public void SetLeftRight(UnitTrace left, UnitTrace right)
        {
            _unitLeft = left;
            _unitRight = right;
        }

        public UnitTrace EndUnit(bool matched)
        {
            return new UnitTrace
            {
                UnitIndex = _unitIndex,
                Mode = _unitMode,
                Optional = _unitOptional,
                TokenIndex = _unitTokenIndex,
                TokenValue = _unitTokenValue,
                Matched = matched,
                Criteria = _unitCriteria ?? new List<CriterionTrace>(),
                LeftSide = _unitLeft,
                RightSide = _unitRight
            };
        }

        public void AddUnitToAlternative(UnitTrace ut)
        {
            (_altUnits ?? (_altUnits = new List<UnitTrace>())).Add(ut);
        }

        // Save/restore scratch so PatternUnit.IsMatch can recurse into And/Or sub-units
        // without losing the enclosing unit's pending state.
        public struct Scratch
        {
            internal int UnitIndex;
            internal PatternMatchingMode UnitMode;
            internal bool UnitOptional;
            internal int UnitTokenIndex;
            internal string UnitTokenValue;
            internal List<CriterionTrace> UnitCriteria;
            internal UnitTrace UnitLeft;
            internal UnitTrace UnitRight;
        }

        public Scratch SaveUnit()
        {
            return new Scratch
            {
                UnitIndex = _unitIndex,
                UnitMode = _unitMode,
                UnitOptional = _unitOptional,
                UnitTokenIndex = _unitTokenIndex,
                UnitTokenValue = _unitTokenValue,
                UnitCriteria = _unitCriteria,
                UnitLeft = _unitLeft,
                UnitRight = _unitRight
            };
        }

        public void RestoreUnit(Scratch s)
        {
            _unitIndex = s.UnitIndex;
            _unitMode = s.UnitMode;
            _unitOptional = s.UnitOptional;
            _unitTokenIndex = s.UnitTokenIndex;
            _unitTokenValue = s.UnitTokenValue;
            _unitCriteria = s.UnitCriteria;
            _unitLeft = s.UnitLeft;
            _unitRight = s.UnitRight;
        }

        public int CurrentUnitTokenIndex => _unitTokenIndex;
        public string CurrentUnitTokenValue => _unitTokenValue;

        // Finalizes outcomes against the pattern's winning consumed-token count.
        // Ties on the winning count are all reported as Won — they are equally valid
        // explanations of the pattern's result.
        public List<AlternativeTrace> Drain(int winnerConsumed)
        {
            var result = new List<AlternativeTrace>(_alternatives.Count);
            foreach (var a in _alternatives)
            {
                AlternativeOutcome outcome;
                if (!a.Completed)
                {
                    outcome = AlternativeOutcome.Failed;
                }
                else if (winnerConsumed > 0 && a.Consumed == winnerConsumed)
                {
                    outcome = AlternativeOutcome.Won;
                }
                else
                {
                    outcome = AlternativeOutcome.LostShorter;
                }

                result.Add(new AlternativeTrace
                {
                    AlternativeIndex = a.Index,
                    Units = a.Units ?? new List<UnitTrace>(),
                    Outcome = outcome,
                    ConsumedTokens = a.Consumed,
                });
            }
            _alternatives.Clear();
            return result;
        }
    }
}
