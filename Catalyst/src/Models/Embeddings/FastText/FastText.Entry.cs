using MessagePack;
using Mosaik.Core;

namespace Catalyst.Models
{
    public partial class FastText
    {
        [MessagePackObject]
        public struct Entry
        {
            [Key(0)] public string Word;
            [Key(1)] public long Count;
            [Key(2)] public EntryType Type;
            [Key(3)] public PartOfSpeech POS;
            [Key(4)] public Language Language;

            [SerializationConstructor]
            public Entry(string word, long count, EntryType type, PartOfSpeech pos, Language language)
            {
                Word = word;
                Count = count;
                Type = type;
                POS = pos;
                Language = language;
            }
        }

        public enum EntryType
        {
            Word,
            Label
        }
    }
}