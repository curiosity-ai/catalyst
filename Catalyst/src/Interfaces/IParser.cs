namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a parser that can perform syntactic parsing on a document or span.
    /// </summary>
    public interface IParser
    {
        /// <summary>
        /// Parses the specified document.
        /// </summary>
        /// <param name="document">The document to parse.</param>
        void Parse(IDocument document);

        /// <summary>
        /// Parses the specified span.
        /// </summary>
        /// <param name="span">The span to parse.</param>
        void Parse(Span span);
    }
}