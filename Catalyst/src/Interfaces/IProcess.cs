using Mosaik.Core;

namespace Catalyst
{
    public interface IProcess : IModel
    {
        void Process(IDocument document);
    }
}