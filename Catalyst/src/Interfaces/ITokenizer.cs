using System.Threading;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a tokenizer that can segment a document or span into tokens.
    /// </summary>
    public interface ITokenizer
    {
        /// <summary>
        /// Tokenizes the specified document.
        /// </summary>
        /// <param name="document">The document to tokenize.</param>
        /// <param name="cancellationToken">A cancellation token to observe while tokenizing the document.</param>
        void Parse(IDocument document, CancellationToken cancellationToken = default);

        /// <summary>
        /// Tokenizes the specified span.
        /// </summary>
        /// <param name="span">The span to tokenize.</param>
        /// <param name="cancellationToken">A cancellation token to observe while tokenizing the span.</param>
        void Parse(Span span, CancellationToken cancellationToken = default);
    }
}