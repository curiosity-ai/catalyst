using Mosaik.Core;

namespace Catalyst
{
    public interface IEntityRecognizer : IModel
    {
        bool RecognizeEntities(IDocument document);

        string[] Produces();
    }
}