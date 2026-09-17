namespace Catalyst.DateTimeRecognition
{
    public ref partial struct Parser
    {
        /// <summary>
        /// Tries every construct at <paramref name="i"/> and keeps the longest. Where two constructs cover the
        /// same span the order below decides, from the most specific reading to the least.
        /// </summary>
        public int TryMatch(int i, out int node)
        {
            int best     = -1;
            int bestNode = Node.Unspecified;

            Consider(TrySetWithLeadingTime(i, out int n0), n0, ref best, ref bestNode);
            Consider(TrySet(i, out int n1),                n1, ref best, ref bestNode);
            Consider(TryDateTimePeriod(i, out int n2),     n2, ref best, ref bestNode);
            Consider(TryDatePeriod(i, out int n3),         n3, ref best, ref bestNode);
            Consider(TryDateTime(i, out int n4),           n4, ref best, ref bestNode);
            Consider(TryTimePeriod(i, out int n5),         n5, ref best, ref bestNode);
            Consider(TryDate(i, out int n6),               n6, ref best, ref bestNode);
            Consider(TryTime(i, out int n7),               n7, ref best, ref bestNode);
            Consider(TryDuration(i, out int n8),           n8, ref best, ref bestNode);

            node = bestNode;
            return best;
        }
    }
}
