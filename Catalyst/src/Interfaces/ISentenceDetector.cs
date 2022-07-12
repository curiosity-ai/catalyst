using System.Threading;

namespace Catalyst
{
    public interface ISentenceDetector
    {
        void Parse(IDocument document, CancellationToken cancellationToken = default);
    }
}