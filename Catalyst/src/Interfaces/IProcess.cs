using MessagePack;
using Mosaik.Core;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a model that can process a document.
    /// </summary>
    public interface IProcess : IModel
    {
        /// <summary>
        /// Processes the specified document.
        /// </summary>
        /// <param name="document">The document to process.</param>
        /// <param name="cancellationToken">A cancellation token to observe while processing the document.</param>
        void Process(IDocument document, CancellationToken cancellationToken = default);
    }
}