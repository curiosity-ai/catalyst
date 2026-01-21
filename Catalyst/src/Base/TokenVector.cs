using MessagePack;
using Mosaik.Core;

namespace Catalyst
{
    /// <summary>
    /// Represents a token and its associated vector representation.
    /// </summary>
    public struct TokenVector
    {
        /// <summary>Gets or sets the token text.</summary>
        [Key(0)] public string Token { get; set; }
        /// <summary>Gets or sets the vector representation.</summary>
        [Key(1)] public float[] Vector { get; set; }
        /// <summary>Gets or sets the hash of the token.</summary>
        [Key(2)] public int Hash { get; set; }
        /// <summary>Gets or sets the part-of-speech tag.</summary>
        [Key(3)] public PartOfSpeech POS;
        /// <summary>Gets or sets the language of the token.</summary>
        [Key(4)] public Language Language;
        /// <summary>Gets or sets the frequency of the token.</summary>
        [Key(5)] public float Frequency;

        /// <summary>
        /// Initializes a new instance of the <see cref="TokenVector"/> struct.
        /// </summary>
        /// <param name="token">The token text.</param>
        /// <param name="vector">The vector representation.</param>
        /// <param name="hash">The token hash.</param>
        /// <param name="pos">The part-of-speech tag.</param>
        /// <param name="language">The language.</param>
        /// <param name="frequency">The frequency.</param>
        public TokenVector(string token, float[] vector, int hash, PartOfSpeech pos, Language language, float frequency) : this()
        {
            Token = token;
            Vector = vector;
            Hash = hash;
            POS = pos;
            Language = language;
            Frequency = frequency;
        }
    }
}