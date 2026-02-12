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
                var tokens = span.Tokens.ToArray();
                for (int i = 0; i < tokens.Length; i++)
                {
                     foreach (var et in tokens[i].EntityTypes)
                     {
                         if (et.Type == "EmailOrURL") continue;

                         if (et.Tag == EntityTag.Single)
                         {
                             var result = new RecognizerResult
                             {
                                 Start = tokens[i].Begin,
                                 End = tokens[i].End,
                                 EntityType = et.Type,
                                 Score = 1.0 // Default score
                             };
                             if (processedEntities.Add((result.Start, result.End, result.EntityType)))
                             {
                                 results.Add(result);
                             }
                         }
                         else if (et.Tag == EntityTag.Begin)
                         {
                             // Find end of entity
                             int j = i + 1;
                             bool foundEnd = false;

                             while (j < tokens.Length)
                             {
                                 var nextToken = tokens[j];
                                 bool foundNext = false;

                                 foreach(var net in nextToken.EntityTypes)
                                 {
                                     if(net.Type == et.Type)
                                     {
                                         if(net.Tag == EntityTag.Inside)
                                         {
                                             foundNext = true;
                                             // Continue searching
                                         }
                                         else if(net.Tag == EntityTag.End)
                                         {
                                             foundNext = true;
                                             foundEnd = true;

                                             var result = new RecognizerResult
                                             {
                                                 Start = tokens[i].Begin,
                                                 End = tokens[j].End,
                                                 EntityType = et.Type,
                                                 Score = 1.0
                                             };
                                              if (processedEntities.Add((result.Start, result.End, result.EntityType)))
                                             {
                                                 results.Add(result);
                                             }
                                             break;
                                         }
                                     }
                                 }

                                 if (foundEnd) break;
                                 if (!foundNext) break; // Chain broken
                                 j++;
                             }
                         }
                     }
                }
            }

            return results;
        }
    }
}
