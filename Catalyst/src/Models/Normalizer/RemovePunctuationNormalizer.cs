using Mosaik.Core;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class RemovePunctuationNormalizer : INormalizer, IProcess
    {
        public Language Language => Language.Any;
        public string Type => typeof(RemovePunctuationNormalizer).FullName;
        public string Tag => "";
        public int Version => 0;

        public static async Task<RemovePunctuationNormalizer> FromStoreAsync(Language language, int version, string tag)
        {
            return await Task.FromResult(new RemovePunctuationNormalizer());
        }

        public static async Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return true;
        } // Needs to say it exists, otherwise when calling ModelDescription.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public void Process(IDocument document)
        {
            Normalize(document);
        }

        //private static Regex RE_MultiHyphens = new Regex($"([{CharacterClasses.Hyphens}]){CharacterClasses.Hyphens}+", RegexOptions.Compiled);
        private static Regex RE_MultiHyphens = new Regex($"[{CharacterClasses.Hyphens}]+(?![0-9])", RegexOptions.Compiled);

        private static char[] Hyphens = CharacterClasses.Hyphens.Split('|').Select(s => s[0]).ToArray();

        public void Normalize(IDocument document)
        {
            foreach (var span in document)
            {
                foreach (var token in span)
                {
                    if (token.ValueAsSpan.IsAnyPunctuation()) { token.Replacement = ""; }    // Remove all punctuation
                    if (token.Value.Length > 1)
                    {
                        if (token.Value.IndexOfAny(Hyphens) > -1)
                        {
                            if (token.ValueAsSpan.IsHyphen())
                            {
                                token.Replacement = "-"; // token.Value.Substring(0,1);  //replace multiple hyphens with single ones
                            }
                            else if (RE_MultiHyphens.IsMatch(token.Value))
                            {
                                token.Replacement = RE_MultiHyphens.Replace(token.Value, "_"); // "$1");
                            }
                        }
                    }
                }
            }
        }

        public string Normalize(string text)
        {
            throw new System.NotImplementedException();
        }
    }
}