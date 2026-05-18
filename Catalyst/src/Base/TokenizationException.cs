using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents a tokenization exception, which defines a custom segmentation for a specific word.
    /// </summary>
    [MessagePackObject]
    public struct TokenizationException
    {
        /// <summary>Gets or sets the sequence of tokens that the word should be split into.</summary>
        [Key(0)] public string[] Replacements;

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenizationException"/> struct.
        /// </summary>
        /// <param name="replacements">The sequence of tokens.</param>
        [SerializationConstructor]
        public TokenizationException(string[] replacements) { Replacements = replacements; }
    }
}