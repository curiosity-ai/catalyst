namespace Catalyst
{
    public interface ISentenceDetector
    {
        void Parse(IDocument document);
    }
}