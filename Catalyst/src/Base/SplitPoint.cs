namespace Catalyst.Models
{

    internal struct SplitPoint
    {
        internal int Begin;
        internal int End;
        internal SplitPointReason Reason;

        internal SplitPoint(int b, int e, SplitPointReason reason)
        {
            Begin = b; End = e; Reason = reason;
        }
    }
}