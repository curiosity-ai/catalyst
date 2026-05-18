using Mosaik.Core;
using System;
using System.Collections.Concurrent;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a lemmatizer.
    /// </summary>
    public interface ILemmatizer
    {
        /// <summary>
        /// Gets the language supported by this lemmatizer.
        /// </summary>
        Language Language { get; }
     
        /// <summary>
        /// Check if the token is an uninflected paradigm, so we can avoid lemmatization entirely.
        /// </summary>
        /// <param name="token">The token to check.</param>
        /// <returns>True if the token is in its base form, false otherwise.</returns>
        bool IsBaseForm(IToken token);
        
        /// <summary>
        /// Gets the lemma for the specified token.
        /// </summary>
        /// <param name="token">The token to lemmatize.</param>
        /// <returns>The lemma of the token.</returns>
        string GetLemma(IToken token);
        
        /// <summary>
        /// Gets the lemma for the specified token as a read-only span of characters.
        /// </summary>
        /// <param name="token">The token to lemmatize.</param>
        /// <returns>A read-only span containing the lemma.</returns>
        ReadOnlySpan<char> GetLemmaAsSpan(IToken token);
    }
}