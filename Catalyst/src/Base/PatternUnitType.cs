using System;

namespace Catalyst
{
    //Don't change any previous existing if adding new item
    [Flags]
    public enum PatternUnitType
    {
        Token                = 0b000000000000000000000000001,
        Shape                = 0b000000000000000000000000010,
        Script               = 0b000000000000000000000000100,
        POS                  = 0b000000000000000000000001000,
        MultiplePOS          = 0b000000000000000000000010000,
        Suffix               = 0b000000000000000000000100000,
        Prefix               = 0b000000000000000000001000000,
        Set                  = 0b000000000000000000010000000,
        Entity               = 0b000000000000000000100000000,
        IsDigit              = 0b000000000000000001000000000,
        IsNumeric            = 0b000000000000000010000000000,
        IsAlpha              = 0b000000000000000100000000000,
        IsLetterOrDigit      = 0b000000000000001000000000000,
        IsLatin              = 0b000000000000010000000000000,
        IsEmoji              = 0b000000000000100000000000000,
        IsPunctuation        = 0b000000000001000000000000000,
        IsLowerCase          = 0b000000000010000000000000000,
        IsUpperCase          = 0b000000000100000000000000000,
        IsTitleCase          = 0b000000001000000000000000000,
        LikeURL              = 0b000000010000000000000000000,
        LikeEmail            = 0b000000100000000000000000000,
        IsOpeningParenthesis = 0b000001000000000000000000000,
        IsClosingParenthesis = 0b000010000000000000000000000,
        NotEntity            = 0b000100000000000000000000000,
        HasNumeric           = 0b001000000000000000000000000,
        WithChars            = 0b010000000000000000000000000,
        Length               = 0b100000000000000000000000000,
    }
}
