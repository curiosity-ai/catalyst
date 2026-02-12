using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using P = Catalyst.PatternUnitPrototype;

namespace Catalyst.Presidio
{
    public class PresidioAnalyzer
    {
        private Pipeline _nlp;
        private Language _language;

        public Language Language => _language;

        private PresidioAnalyzer(Language language)
        {
            _language = language;
            _nlp = new Pipeline();
            _nlp.Add(new FastTokenizer(language));
        }

        public static PresidioAnalyzer For(Language language)
        {
            return new PresidioAnalyzer(language);
        }

        public PresidioAnalyzer AddRecognizer(IProcess model)
        {
            _nlp.Add(model);
            return this;
        }

        public List<RecognizerResult> Analyze(string text)
        {
            var doc = new Document(text, _language);
            _nlp.ProcessSingle(doc);

            var results = new List<RecognizerResult>();
            var processedEntities = new HashSet<(int, int, string)>();

            foreach (var span in doc.Spans)
            {
                foreach (var entity in span.GetEntities())
                {
                    if (entity.EntityType.Type == "EmailOrURL") continue;

                    var result = new RecognizerResult
                    {
                        Start = entity.Begin,
                        End = entity.End,
                        EntityType = entity.EntityType.Type,
                        Score = 1.0 // Default score
                    };

                    if (processedEntities.Add((result.Start, result.End, result.EntityType)))
                    {
                        results.Add(result);
                    }
                }
            }

            return results;
        }
    }
}
