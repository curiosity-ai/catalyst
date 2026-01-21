using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a model that provides vector representations (embeddings) for tokens.
    /// </summary>
    public interface IVectorModel
    {
        /// <summary>
        /// Returns an enumerable of all token vectors in the model.
        /// </summary>
        /// <returns>An enumerable of token vectors.</returns>
        IEnumerable<TokenVector> GetTokenVectors();

        /// <summary>
        /// Gets the vector representation for the specified token.
        /// </summary>
        /// <param name="token">The token text.</param>
        /// <returns>The token vector.</returns>
        TokenVector GetTokenVector(string token);

        /// <summary>
        /// Gets the vector representations for the specified tokens.
        /// </summary>
        /// <param name="tokens">The collection of token texts.</param>
        /// <returns>An enumerable of token vectors.</returns>
        IEnumerable<TokenVector> GetTokenVector(IEnumerable<string> tokens);

        /// <summary>
        /// Gets the k most similar tokens to the specified token vector.
        /// </summary>
        /// <param name="token">The token vector.</param>
        /// <param name="k">The number of similar tokens to return.</param>
        /// <returns>An enumerable of the most similar tokens.</returns>
        IEnumerable<MostSimilar> GetMostSimilar(TokenVector token, int k);

        /// <summary>
        /// Asynchronously gets the k most similar tokens to the specified token vector.
        /// </summary>
        /// <param name="token">The token vector.</param>
        /// <param name="k">The number of similar tokens to return.</param>
        /// <returns>A task that represents the asynchronous operation, containing an enumerable of the most similar tokens.</returns>
        Task<IEnumerable<MostSimilar>> GetMostSimilarAsync(TokenVector token, int k);
    }
}