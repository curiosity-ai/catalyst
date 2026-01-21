using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Catalyst
{
    /// <summary>
    /// Represents a prototype for a pattern unit, providing a concrete implementation of <see cref="IPatternUnit"/>.
    /// </summary>
    public class PatternUnitPrototype : IPatternUnit
    {
        /// <summary>Gets or sets the matching mode.</summary>
        public PatternMatchingMode Mode { get; set; }

        /// <summary>Gets or sets a value indicating whether matching this unit is optional.</summary>
        public bool Optional { get; set; }

        /// <summary>Gets or sets a value indicating whether matching is case-sensitive.</summary>
        public bool CaseSensitive { get; set; }

        /// <summary>Gets or sets the type of pattern unit.</summary>
        public PatternUnitType Type { get; set; }

        /// <summary>Gets or sets the required part-of-speech tags.</summary>
        public PartOfSpeech[] POS { get; set; }

        /// <summary>Gets or sets the required suffix.</summary>
        public string Suffix { get; set; }

        /// <summary>Gets or sets the required prefix.</summary>
        public string Prefix { get; set; }

        /// <summary>Gets or sets the required character shape.</summary>
        public string Shape { get; set; }

        /// <summary>Gets or sets the required token text.</summary>
        public string Token { get; set; }

        /// <summary>Gets or sets the set of valid token texts.</summary>
        public HashSet<string> Set { get; set; }

        /// <summary>Gets or sets the required entity type.</summary>
        public string EntityType { get; set; }

        /// <summary>Gets or sets the hashes of valid token texts in the set.</summary>
        public HashSet<ulong> SetHashes { get; set; }

        /// <summary>Gets or sets the hash of the required token text.</summary>
        public ulong TokenHash { get; set; }

        /// <summary>Gets or sets the left side of a binary operation (AND/OR).</summary>
        public PatternUnitPrototype LeftSide { get; set; }

        /// <summary>Gets or sets the right side of a binary operation (AND/OR).</summary>
        public PatternUnitPrototype RightSide { get; set; }

        /// <summary>Gets or sets the set of valid characters.</summary>
        public HashSet<char> ValidChars { get; set; }

        /// <summary>Gets or sets the minimum required length.</summary>
        public int MinLength { get; set; }

        /// <summary>Gets or sets the maximum required length.</summary>
        public int MaxLength { get; set; }

        /// <summary>Gets or sets the maximum number of matches for multiple matching.</summary>
        public int MaxMatches { get; set; }

        /// <summary>Creates a pattern unit that matches exactly once.</summary>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit Single()           { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Single }; }

        /// <summary>Creates a pattern unit that matches one or more times.</summary>
        /// <param name="maxMatches">The maximum number of matches.</param>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit Multiple(int maxMatches = 10) { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Multiple, MaxMatches = maxMatches }; }

        /// <summary>Creates a pattern unit that should not match.</summary>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit ShouldNotMatch()   { return new PatternUnitPrototype() { Mode = PatternMatchingMode.ShouldNotMatch }; }

        /// <summary>Creates a pattern unit that matches at most once (optional).</summary>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit SingleOptional()   { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Single, Optional = true }; }

        /// <summary>Creates a pattern unit that matches zero or more times (optional multiple).</summary>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit MultipleOptional() { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Multiple, Optional = true }; }

        /// <summary>Creates a pattern unit that matches if both specified units match.</summary>
        /// <param name="A">The first unit.</param>
        /// <param name="B">The second unit.</param>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit And(IPatternUnit A, IPatternUnit B) { return new PatternUnitPrototype() { Mode = PatternMatchingMode.And, LeftSide = (PatternUnitPrototype)A, RightSide = (PatternUnitPrototype)B }; }

        /// <summary>Creates a pattern unit that matches if either of the specified units match.</summary>
        /// <param name="A">The first unit.</param>
        /// <param name="B">The second unit.</param>
        /// <returns>A new <see cref="IPatternUnit"/>.</returns>
        public static IPatternUnit Or (IPatternUnit A, IPatternUnit B) { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Or,  LeftSide = (PatternUnitPrototype)A, RightSide = (PatternUnitPrototype)B }; }

        /// <summary>Combines two 64-bit hashes.</summary>
        /// <param name="rhs">The right-hand side hash.</param>
        /// <param name="lhs">The left-hand side hash.</param>
        /// <returns>The combined hash.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong HashCombine64(ulong rhs, ulong lhs)
        {
            lhs ^= rhs + 0x9e3779b97f492000 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }

        /// <summary>Computes a 64-bit hash for the specified character span.</summary>
        /// <param name="key">The character span.</param>
        /// <returns>The computed hash.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Hash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += key[i];
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        /// <summary>Computes a case-insensitive 64-bit hash for the specified character span.</summary>
        /// <param name="key">The character span.</param>
        /// <returns>The computed hash.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong IgnoreCaseHash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += char.ToLowerInvariant(key[i]);
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        /// <inheritdoc />
        public IPatternUnit WithToken(string token, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Token;
            Token = token;
            CaseSensitive = !ignoreCase;
            TokenHash = ignoreCase ? IgnoreCaseHash64(token.AsSpan()) : Hash64(token.AsSpan());
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithTokens(IEnumerable<string> tokens, bool ignoreCase = false)
        {
            if (tokens.Count() == 1)
            {
                return WithToken(tokens.First(), ignoreCase);
            }
            Type |= PatternUnitType.Set;
            Set = new HashSet<string>(tokens);
            CaseSensitive = !ignoreCase;
            SetHashes = new HashSet<ulong>(tokens.Select(tk => ignoreCase ? IgnoreCaseHash64(tk.AsSpan()) : Hash64(tk.AsSpan())));
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithChars(IEnumerable<char> chars)
        {
            ValidChars = new HashSet<char>(chars);
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithPOS(params PartOfSpeech[] pos)
        {
            Type |= pos.Length > 1 ? PatternUnitType.MultiplePOS : PatternUnitType.POS;
            POS = pos;
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithPrefix(string prefix, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Prefix;
            Prefix = prefix;
            CaseSensitive = !ignoreCase;
            return this;
        }
        /// <inheritdoc />
        public IPatternUnit WithSuffix(string suffix, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Suffix;
            Suffix = suffix;
            CaseSensitive = !ignoreCase;
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithEntityType(string entityType)
        {
            Type |= PatternUnitType.Entity;
            EntityType = entityType;
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithoutEntityType(string entityType)
        {
            Type |= PatternUnitType.NotEntity;
            EntityType = entityType;
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithShape(string shape)
        {
            Type |= PatternUnitType.Shape;
            Shape = string.Join(",", shape.Split(PatternUnit.splitChar, StringSplitOptions.RemoveEmptyEntries).Select(s => s.AsSpan().Shape(false)));
            return this;
        }

        /// <inheritdoc />
        public IPatternUnit WithLength(int minLength, int maxLength)
        {
            Type |= PatternUnitType.Length;
            MinLength = minLength;
            MaxLength = maxLength;
            return this;
        }
        /// <inheritdoc />
        public IPatternUnit IsDigit() { Type |= PatternUnitType.IsDigit; return this; }
        /// <inheritdoc />
        public IPatternUnit IsNumeric() { Type |= PatternUnitType.IsNumeric; return this; }
        /// <inheritdoc />
        public IPatternUnit HasNumeric() { Type |= PatternUnitType.HasNumeric; return this; }
        /// <inheritdoc />
        public IPatternUnit IsAlpha() { Type |= PatternUnitType.IsAlpha; return this; }
        /// <inheritdoc />
        public IPatternUnit IsLetterOrDigit() { Type |= PatternUnitType.IsLetterOrDigit; return this; }
        /// <inheritdoc />
        public IPatternUnit IsEmoji() { Type |= PatternUnitType.IsEmoji; return this; }
        /// <inheritdoc />
        public IPatternUnit IsPunctuation() { Type |= PatternUnitType.IsPunctuation; return this; }
        /// <inheritdoc />
        public IPatternUnit IsLowerCase() { Type |= PatternUnitType.IsLowerCase; return this; }
        /// <inheritdoc />
        public IPatternUnit IsUpperCase() { Type |= PatternUnitType.IsUpperCase; return this; }
        /// <inheritdoc />
        public IPatternUnit IsTitleCase() { Type |= PatternUnitType.IsTitleCase; return this; }
        /// <inheritdoc />
        public IPatternUnit LikeURL() { Type |= PatternUnitType.LikeURL; return this; }
        /// <inheritdoc />
        public IPatternUnit LikeEmail() { Type |= PatternUnitType.LikeEmail; return this; }
        /// <inheritdoc />
        public IPatternUnit IsOpeningParenthesis() { Type |= PatternUnitType.IsOpeningParenthesis; return this; }
        /// <inheritdoc />
        public IPatternUnit IsClosingParenthesis() { Type |= PatternUnitType.IsClosingParenthesis; return this; }
    }
}
