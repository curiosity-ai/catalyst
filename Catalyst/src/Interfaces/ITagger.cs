namespace Catalyst
{
    public interface ITagger
    {
        void Predict(IDocument document);

        void Predict(Span span);
    }
}