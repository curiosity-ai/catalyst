namespace Catalyst
{
    public interface IParser
    {
        void Parse(IDocument document);

        void Parse(ISpan span);
    }
}