using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Catalyst
{
    public class PatternUnitPrototype : IPatternUnit
    {
        public PatternMatchingMode Mode;
        public bool Optional;
        public bool CaseSensitive;
        public PatternUnitType Type;
        public PartOfSpeech[] POS;
        public string Suffix;
        public string Prefix;
        public string Shape;
        public string Token;
        public HashSet<string> Set;
        public string EntityType;
        public HashSet<ulong> SetHashes;
        public ulong TokenHash;
        public PatternUnitPrototype LeftSide;
        public PatternUnitPrototype RightSide;
        public HashSet<char> ValidChars;
        public int MinLength;
        public int MaxLength;
        public static IPatternUnit Single()           { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Single }; }
        public static IPatternUnit Multiple()         { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Multiple }; }
        public static IPatternUnit ShouldNotMatch()   { return new PatternUnitPrototype() { Mode = PatternMatchingMode.ShouldNotMatch }; }
        public static IPatternUnit SingleOptional()   { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Single, Optional = true }; }
        public static IPatternUnit MultipleOptional() { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Multiple, Optional = true }; }

        public static IPatternUnit And(IPatternUnit A, IPatternUnit B) { return new PatternUnitPrototype() { Mode = PatternMatchingMode.And, LeftSide = (PatternUnitPrototype)A, RightSide = (PatternUnitPrototype)B }; }
        public static IPatternUnit Or (IPatternUnit A, IPatternUnit B) { return new PatternUnitPrototype() { Mode = PatternMatchingMode.Or,  LeftSide = (PatternUnitPrototype)A, RightSide = (PatternUnitPrototype)B }; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong HashCombine64(ulong rhs, ulong lhs)
        {
            lhs ^= rhs + 0x9e3779b97f492000 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }

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

        public IPatternUnit WithToken(string token, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Token;
            Token = token;
            CaseSensitive = !ignoreCase;
            TokenHash = ignoreCase ? IgnoreCaseHash64(token.AsSpan()) : Hash64(token.AsSpan());
            return this;
        }

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

        public IPatternUnit WithChars(IEnumerable<char> chars)
        {
            ValidChars = new HashSet<char>(chars);
            return this;
        }

        public IPatternUnit WithPOS(params PartOfSpeech[] pos)
        {
            Type |= pos.Length > 1 ? PatternUnitType.MultiplePOS : PatternUnitType.POS;
            POS = pos;
            return this;
        }

        public IPatternUnit WithPrefix(string prefix, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Prefix;
            Prefix = prefix;
            CaseSensitive = !ignoreCase;
            return this;
        }
        public IPatternUnit WithSuffix(string suffix, bool ignoreCase = false)
        {
            Type |= PatternUnitType.Suffix;
            Suffix = suffix;
            CaseSensitive = !ignoreCase;
            return this;
        }

        public IPatternUnit WithEntityType(string entityType)
        {
            Type |= PatternUnitType.Entity;
            EntityType = entityType;
            return this;
        }

        public IPatternUnit WithoutEntityType(string entityType)
        {
            Type |= PatternUnitType.NotEntity;
            EntityType = entityType;
            return this;
        }

        public IPatternUnit WithShape(string shape)
        {
            Type |= PatternUnitType.Shape;
            Shape = shape.AsSpan().Shape(false);
            return this;
        }

        public IPatternUnit WithLength(int minLength, int maxLength)
        {
            Type |= PatternUnitType.Length;
            MinLength = minLength;
            MaxLength = maxLength;
            return this;
        }
        public IPatternUnit IsDigit() { Type |= PatternUnitType.IsDigit; return this; }
        public IPatternUnit IsNumeric() { Type |= PatternUnitType.IsNumeric; return this; }
        public IPatternUnit HasNumeric() { Type |= PatternUnitType.HasNumeric; return this; }
        public IPatternUnit IsAlpha() { Type |= PatternUnitType.IsAlpha; return this; }
        public IPatternUnit IsLetterOrDigit() { Type |= PatternUnitType.IsLetterOrDigit; return this; }
        public IPatternUnit IsEmoji() { Type |= PatternUnitType.IsEmoji; return this; }
        public IPatternUnit IsPunctuation() { Type |= PatternUnitType.IsPunctuation; return this; }
        public IPatternUnit IsLowerCase() { Type |= PatternUnitType.IsLowerCase; return this; }
        public IPatternUnit IsUpperCase() { Type |= PatternUnitType.IsUpperCase; return this; }
        public IPatternUnit IsTitleCase() { Type |= PatternUnitType.IsTitleCase; return this; }
        public IPatternUnit LikeURL() { Type |= PatternUnitType.LikeURL; return this; }
        public IPatternUnit LikeEmail() { Type |= PatternUnitType.LikeEmail; return this; }
        public IPatternUnit IsOpeningParenthesis() { Type |= PatternUnitType.IsOpeningParenthesis; return this; }
        public IPatternUnit IsClosingParenthesis() { Type |= PatternUnitType.IsClosingParenthesis; return this; }
    }
}
