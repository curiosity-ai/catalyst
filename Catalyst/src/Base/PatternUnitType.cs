using System;

namespace Catalyst
{
    /// <summary>
    /// Specifies the type of constraints in a pattern unit.
    /// </summary>
    //Don't change any previous existing if adding new item
    [Flags]
    public enum PatternUnitType
    {
        /// <summary>Matches a specific token text.</summary>
        Token                = 0b0000000000000000000000000001,
        /// <summary>Matches a specific character shape.</summary>
        Shape                = 0b0000000000000000000000000010,
        /// <summary>Matches a specific script (unused).</summary>
        Script               = 0b0000000000000000000000000100,
        /// <summary>Matches a specific part-of-speech tag.</summary>
        POS                  = 0b0000000000000000000000001000,
        /// <summary>Matches multiple part-of-speech tags.</summary>
        MultiplePOS          = 0b0000000000000000000000010000,
        /// <summary>Matches a specific suffix.</summary>
        Suffix               = 0b0000000000000000000000100000,
        /// <summary>Matches a specific prefix.</summary>
        Prefix               = 0b0000000000000000000001000000,
        /// <summary>Matches any token text from a set.</summary>
        Set                  = 0b0000000000000000000010000000,
        /// <summary>Matches a specific entity type.</summary>
        Entity               = 0b0000000000000000000100000000,
        /// <summary>Matches if the token consists of digits.</summary>
        IsDigit              = 0b0000000000000000001000000000,
        /// <summary>Matches if the token is numeric.</summary>
        IsNumeric            = 0b0000000000000000010000000000,
        /// <summary>Matches if the token is alphabetic.</summary>
        IsAlpha              = 0b0000000000000000100000000000,
        /// <summary>Matches if the token consists of letters or digits.</summary>
        IsLetterOrDigit      = 0b0000000000000001000000000000,
        /// <summary>Matches if the token is Latin (unused).</summary>
        IsLatin              = 0b0000000000000010000000000000,
        /// <summary>Matches if the token is an emoji.</summary>
        IsEmoji              = 0b0000000000000100000000000000,
        /// <summary>Matches if the token is punctuation.</summary>
        IsPunctuation        = 0b0000000000001000000000000000,
        /// <summary>Matches if the token is lower case.</summary>
        IsLowerCase          = 0b0000000000010000000000000000,
        /// <summary>Matches if the token is upper case.</summary>
        IsUpperCase          = 0b0000000000100000000000000000,
        /// <summary>Matches if the token is in title case.</summary>
        IsTitleCase          = 0b0000000001000000000000000000,
        /// <summary>Matches if the token looks like a URL.</summary>
        LikeURL              = 0b0000000010000000000000000000,
        /// <summary>Matches if the token looks like an email.</summary>
        LikeEmail            = 0b0000000100000000000000000000,
        /// <summary>Matches if the token is an opening parenthesis.</summary>
        IsOpeningParenthesis = 0b0000001000000000000000000000,
        /// <summary>Matches if the token is a closing parenthesis.</summary>
        IsClosingParenthesis = 0b0000010000000000000000000000,
        /// <summary>Matches if the token does not have the specified entity type.</summary>
        NotEntity            = 0b0000100000000000000000000000,
        /// <summary>Matches if the token contains at least one digit.</summary>
        HasNumeric           = 0b0001000000000000000000000000,
        /// <summary>Matches if the token consists of specified characters.</summary>
        WithChars            = 0b0010000000000000000000000000,
        /// <summary>Matches if the token's length is within range.</summary>
        Length               = 0b0100000000000000000000000000,
        /// <summary>Matches a specific token text up to a certain confidence level.</summary>
        Fuzzy                = 0b1000000000000000000000000000,
    }
}
