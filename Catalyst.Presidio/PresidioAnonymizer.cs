using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace Catalyst.Presidio
{
    public class PresidioAnonymizer
    {
        public string Anonymize(string text, List<RecognizerResult> results, string operatorName = "mask", string maskChar = "*", string replacement = "<REDACTED>")
        {
            if (string.IsNullOrEmpty(text)) return text;
            if (results == null || results.Count == 0) return text;

            // Filter overlaps
            var filteredResults = RemoveOverlaps(results);

            // Sort results descending by start index to replace from end to avoid shifting indices
            var sortedResults = filteredResults.OrderByDescending(r => r.Start).ToList();
            var sb = new StringBuilder(text);

            foreach (var result in sortedResults)
            {
                // Validate bounds
                if (result.Start < 0 || result.End >= text.Length || result.Start > result.End) continue;

                var length = result.Length;
                string newText = "";

                switch (operatorName.ToLowerInvariant())
                {
                    case "mask":
                        if (!string.IsNullOrEmpty(maskChar))
                        {
                            newText = new string(maskChar[0], length);
                        }
                        else
                        {
                            newText = new string('*', length);
                        }
                        break;
                    case "replace":
                        newText = replacement;
                        break;
                    case "redact":
                        newText = "";
                        break;
                    case "hash":
                        var original = text.Substring(result.Start, length);
                        using (var sha256 = SHA256.Create())
                        {
                            var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(original));
                            newText = BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
                        }
                        break;
                    default:
                        // Default to mask
                        newText = new string('*', length);
                        break;
                }

                sb.Remove(result.Start, length);
                sb.Insert(result.Start, newText);
            }

            return sb.ToString();
        }

        private List<RecognizerResult> RemoveOverlaps(List<RecognizerResult> results)
        {
            // Sort by start position
            var sorted = results.OrderBy(r => r.Start).ThenByDescending(r => r.Score).ThenByDescending(r => r.Length).ToList();
            var output = new List<RecognizerResult>();

            int lastEnd = -1;

            foreach (var result in sorted)
            {
                if (result.Start > lastEnd)
                {
                    output.Add(result);
                    lastEnd = result.End;
                }
                else
                {
                    // Overlap. Since we sorted by Score/Length descending, we keep the previous one and drop this one.
                    // Or if this one extends beyond lastEnd?
                    // Simple logic: strictly no overlap allowed.
                    // If we want to support nested entities (e.g. replace inner, then replace outer), it's complicated.
                    // Presidio usually drops overlapping entities for anonymization to avoid corruption.
                }
            }
            return output;
        }
    }
}
