using Mosaik.Core;
using System;
using System.Collections.Concurrent;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a lemmatizer.
    /// </summary>
    /// <remarks>
    /// Lemmatizing takes the token's value rather than the token: <see cref="Token"/> is a struct, so a
    /// lemmatizer that asked for an <see cref="IToken"/> boxed one on every call - and a search index asks once
    /// per token it writes. Every implementation only ever read the value, so nothing was lost in narrowing it.
    /// </remarks>
    public interface ILemmatizer
    {
        /// <summary>
        /// Gets the language supported by this lemmatizer.
        /// </summary>
        Language Language { get; }
     
        /// <summary>
        /// Check if the value is an uninflected paradigm, so we can avoid lemmatization entirely.
        /// </summary>
        /// <param name="value">The token's value.</param>
        /// <returns>True if the value is in its base form, false otherwise.</returns>
        bool IsBaseForm(ReadOnlySpan<char> value);

        /// <summary>
        /// Gets the lemma for the specified token value.
        /// </summary>
        /// <param name="value">The token's value.</param>
        /// <returns>The lemma of the value.</returns>
        string GetLemma(ReadOnlySpan<char> value);

        /// <summary>
        /// Gets the lemma for the specified token value as a read-only span of characters.
        /// </summary>
        /// <param name="value">The token's value.</param>
        /// <returns>A read-only span containing the lemma.</returns>
        ReadOnlySpan<char> GetLemmaAsSpan(ReadOnlySpan<char> value);
    }
}