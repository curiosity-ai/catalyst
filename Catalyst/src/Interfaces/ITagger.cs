using System.Threading;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a part-of-speech tagger.
    /// </summary>
    public interface ITagger
    {
        /// <summary>
        /// Predicts part-of-speech tags for tokens in the specified document.
        /// </summary>
        /// <param name="document">The document to process.</param>
        /// <param name="cancellationToken">A cancellation token to observe while processing the document.</param>
        void Predict(IDocument document, CancellationToken cancellationToken = default);

        /// <summary>
        /// Predicts part-of-speech tags for tokens in the specified span.
        /// </summary>
        /// <param name="span">The span to process.</param>
        void Predict(Span span);
    }
}