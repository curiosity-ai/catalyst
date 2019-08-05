using MessagePack;

namespace Catalyst.Models
{
    public partial class FastText
    {
        [MessagePackObject]
        public struct Line
        {
            [Key(0)] public int[] EntryIndexes;
            [Key(1)] public int[] Labels;

            [SerializationConstructor]
            public Line(int[] entryIndexes, int[] labels)
            {
                EntryIndexes = entryIndexes;
                Labels = labels;
            }
        }
    }
}