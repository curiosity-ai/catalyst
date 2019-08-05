namespace Catalyst
{
    public interface ITokenizer
    {
        void Parse(IDocument document);

        void Parse(ISpan span);
    }
}