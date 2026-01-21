using System.Threading;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a sentence detector that can segment a document into sentences.
    /// </summary>
    public interface ISentenceDetector
    {
        /// <summary>
        /// Segments the specified document into sentences (spans).
        /// </summary>
        /// <param name="document">The document to process.</param>
        /// <param name="cancellationToken">A cancellation token to observe while processing the document.</param>
        void Parse(IDocument document, CancellationToken cancellationToken = default);
    }
}