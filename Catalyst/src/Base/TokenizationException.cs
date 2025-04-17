using MessagePack;

namespace Catalyst
{
    [MessagePackObject]
    public partial struct TokenizationException
    {
        [Key(0)] public string[] Replacements;

        [SerializationConstructor]
        public TokenizationException(string[] replacements) { Replacements = replacements; }
    }
}