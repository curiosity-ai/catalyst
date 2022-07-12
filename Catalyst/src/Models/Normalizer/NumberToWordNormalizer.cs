using Mosaik.Core;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class NumberToWordNormalizer : INormalizer, IProcess
    {
        public Language Language => Language.Any;
        public string Type => typeof(NumberToWordNormalizer).FullName;
        public string Tag => "";
        public int Version => 0;

        public string ReplacementText { get; set; } = "_number_";

        public static async Task<NumberToWordNormalizer> FromStoreAsync(Language language, int version, string tag)
        {
            return await Task.FromResult(new NumberToWordNormalizer());
        }

        public static async Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return true;
        } // Needs to say it exists, otherwise when calling ModelDescription.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Normalize(document);
        }

        public void Normalize(IDocument document)
        {
            foreach (var span in document)
            {
                foreach (var token in span)
                {
                    if (token.ValueAsSpan.IsNumeric()) { token.Replacement = ReplacementText; }
                }
            }
        }

        public string Normalize(string text)
        {
            throw new NotImplementedException();
        }
    }
}