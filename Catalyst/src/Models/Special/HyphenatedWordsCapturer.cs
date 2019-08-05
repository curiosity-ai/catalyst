using Mosaik.Core;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class HyphenatedWordsCapturer : IEntityRecognizer, IProcess
    {
        public Language Language => Language.Any;
        public string Type => typeof(HyphenatedWordsCapturer).FullName;

        public string Tag => "";
        public int Version => 0;

        public static async Task<HyphenatedWordsCapturer> FromStoreAsync(Language language, int version, string tag)
        {
            return await Task.FromResult(new HyphenatedWordsCapturer());
        }

        public static async Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return true;
        } // Needs to say it exists, otherwise when calling ModelDescription.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public void Process(IDocument document)
        {
            RecognizeEntities(document);
        }

        public bool RecognizeEntities(IDocument document)
        {
            bool result = false;
            foreach (var span in document)
            {
                result |= RecognizeEntities(span);
            }
            return result;
        }

        public const string HyphenatedTag = "HyphenatedWord";

        public string[] Produces()
        {
            return new[] { HyphenatedTag };
        }

        private bool RecognizeEntities(ISpan span)
        {
            IToken prev = SpecialToken.BeginToken; IToken prev2 = SpecialToken.BeginToken; IToken curr = SpecialToken.BeginToken; IToken next = SpecialToken.BeginToken; IToken next2 = SpecialToken.BeginToken;
            bool prevH = false, prev2H = false, currH = false, nextH = false, next2H = false;
            bool prevP = false, prev2P = false, currP = false, nextP = false, next2P = false;
            bool foundAny = false;

            var en = span.GetEnumerator();

            while (next2 != SpecialToken.EndToken)
            {
                prev2 = prev; prev = curr; curr = next; next = next2;
                prev2H = prevH; prevH = currH; currH = nextH; nextH = next2H;
                prev2P = prevP; prevP = currP; currP = nextP; nextP = next2P;
                if (en.MoveNext()) { next2 = en.Current; next2H = next2.ValueAsSpan.IsHyphen(); next2P = next2.ValueAsSpan.IsAnyPunctuation(); } else { next2 = SpecialToken.EndToken; next2H = false; }
                if (prev != SpecialToken.BeginToken)
                {
                    if (!prevH && currH && !nextH && !prevP && !nextP) // pattern: word <hyphen> word, where word != punctuation
                    {
                        int ix = prev.EntityTypes.FindIndex(0, et => et.Type == HyphenatedTag);
                        if (ix > -1)
                        {
                            var newET = new EntityType(HyphenatedTag, EntityTag.Inside);
                            prev.UpdateEntityType(ix, ref newET);
                        }
                        else
                        {
                            prev.AddEntityType(new EntityType(HyphenatedTag, EntityTag.Begin));
                        }

                        curr.AddEntityType(new EntityType(HyphenatedTag, EntityTag.Inside));
                        next.AddEntityType(new EntityType(HyphenatedTag, EntityTag.End));
                        foundAny = true;
                    }
                }
            }
            return foundAny;
        }
    }
}