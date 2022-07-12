using System.Threading;

namespace Catalyst
{
    public interface ITokenizer
    {
        void Parse(IDocument document, CancellationToken cancellationToken = default);

        void Parse(Span span, CancellationToken cancellationToken = default);
    }
}