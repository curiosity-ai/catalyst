using System.Threading;

namespace Catalyst
{
    public interface ITagger
    {
        void Predict(IDocument document, CancellationToken cancellationToken = default);

        void Predict(Span span);
    }
}