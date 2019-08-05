namespace Catalyst
{
    public interface ITagger
    {
        void Predict(IDocument document);

        void Predict(ISpan span);
    }
}