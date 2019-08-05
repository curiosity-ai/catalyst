namespace Catalyst.Models
{
    internal enum SplitPointReason
    {
        Normal,
        Prefix,
        Sufix,
        Infix,
        SingleChar,
        EmailOrUrl,
        Exception,
        Punctuation,
        Emoji
    }
}