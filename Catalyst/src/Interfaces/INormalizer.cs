namespace Catalyst
{
    public interface INormalizer
    {
        void Normalize(IDocument document);
        string Normalize(string text);
    }
}