namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a normalizer that can normalize document text.
    /// </summary>
    public interface INormalizer
    {
        /// <summary>
        /// Normalizes the specified document.
        /// </summary>
        /// <param name="document">The document to normalize.</param>
        void Normalize(IDocument document);

        /// <summary>
        /// Normalizes the specified text.
        /// </summary>
        /// <param name="text">The text to normalize.</param>
        /// <returns>The normalized text.</returns>
        string Normalize(string text);
    }
}