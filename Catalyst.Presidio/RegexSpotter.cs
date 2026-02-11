using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Presidio
{
    public class RegexSpotterModel : StorableObjectData
    {
        public string Pattern { get; set; }
        public string EntityType { get; set; }
    }

    public class RegexSpotter : StorableObject<RegexSpotter, RegexSpotterModel>, IEntityRecognizer, IProcess
    {
        private Regex _regex;

        public RegexSpotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public RegexSpotter(string pattern, string entityType, Language language = Language.Any, string tag = "") : base(language, 0, tag, compress: false)
        {
            Data.Pattern = pattern;
            Data.EntityType = entityType;
            _regex = new Regex(pattern, RegexOptions.Compiled | RegexOptions.IgnoreCase);
        }

        public static new async Task<RegexSpotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new RegexSpotter(language, version, tag);
            await a.LoadDataAsync();
            a._regex = new Regex(a.Data.Pattern, RegexOptions.Compiled | RegexOptions.IgnoreCase);
            return a;
        }

        public string[] Produces()
        {
            return new[] { Data.EntityType };
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            RecognizeEntities(document);
        }

        public bool RecognizeEntities(IDocument document)
        {
            var text = document.Value;
            var matches = _regex.Matches(text);
            bool foundAny = false;
            var entityType = Data.EntityType;

            foreach (Match match in matches)
            {
                if (!match.Success) continue;

                foundAny = true;
                int start = match.Index;
                int end = match.Index + match.Length - 1;

                // Find tokens overlapping with the match
                // We assume tokens are sorted by position
                // We iterate spans then tokens

                foreach(var span in document.Spans)
                {
                    if (span.End < start) continue;
                    if (span.Begin > end) break; // Spans are ordered? Usually yes.

                    foreach(var token in span.Tokens)
                    {
                        // Check overlap
                        int tokenStart = token.Begin;
                        int tokenEnd = token.End;

                        if (tokenEnd < start) continue;
                        if (tokenStart > end) break; // Tokens are ordered

                        // Overlap found
                        // We mark the token with the entity type.
                        // Tagging strategy:
                        // If token is fully inside match:
                        //   If it's the first token in match -> Begin
                        //   If it's the last token in match -> End
                        //   If it's the only token -> Single
                        //   Else -> Inside
                        // If token partially overlaps -> Usually imply bad tokenization for this entity,
                        // but we mark it as part of entity anyway?
                        // Presidio logic would be character based.
                        // Catalyst logic is token based.
                        // We will mark any overlapping token.

                        EntityTag tag = EntityTag.Inside;

                        // Determine tag based on position relative to match start/end
                        // This logic is tricky if token boundaries don't align with match boundaries.
                        // Simple approach:
                        // If token start matches match start -> Begin/Single
                        // If token end matches match end -> End/Single

                        bool isBegin = tokenStart == start;
                        bool isEnd = tokenEnd == end;

                        if (isBegin && isEnd) tag = EntityTag.Single;
                        else if (isBegin) tag = EntityTag.Begin;
                        else if (isEnd) tag = EntityTag.End;
                        else tag = EntityTag.Inside;

                        token.AddEntityType(new EntityType(entityType, tag));
                    }
                }
            }
            return foundAny;
        }
    }
}
