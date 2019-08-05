namespace Catalyst
{
    public interface IMatchingPattern
    {
        IMatchingPattern Add(params IPatternUnit[] units);
    }
}