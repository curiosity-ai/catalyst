namespace Catalyst.Models
{
    /// <summary>
    /// A process that needs some words kept whole by the tokenizer, identified by their case-sensitive
    /// 32-bit hash. The table is handed over by reference - the tokenizer holds onto it rather than copying
    /// it - so a model with millions of exceptions is not duplicated per pipeline it is added to.
    /// </summary>
    internal interface IHasSimpleSpecialCases
    {
        CompactHash32Set GetSimpleSpecialCases();
    }
}
