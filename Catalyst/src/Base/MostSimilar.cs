using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents a token and its similarity score.
    /// </summary>
    public struct MostSimilar
    {
        /// <summary>Gets or sets the token vector.</summary>
        [Key(0)] public TokenVector Token { get; set; }

        /// <summary>Gets or sets the similarity score.</summary>
        [Key(1)] public float Score { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MostSimilar"/> struct.
        /// </summary>
        /// <param name="token">The token vector.</param>
        /// <param name="score">The similarity score.</param>
        public MostSimilar(TokenVector token, float score)
        {
            Token = token; Score = score;
        }
    }
}