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
        private Language _language = Language.English;

        public PresidioAnalyzer()
        {
            _nlp = new Pipeline();
        }

        public async Task InitializeAsync(Language language = Language.English)
        {
            _language = language;
            // Add Tokenizer
            var tokenizer = new FastTokenizer(language);
            _nlp.Add(tokenizer);

            // Add Email Spotter
            var emailSpotter = new PatternSpotter(language, 0, "email", "EMAIL_ADDRESS");
            emailSpotter.NewPattern("Email", mp => mp.Add(new PatternUnit(P.Single().LikeEmail())));
            _nlp.Add(emailSpotter);

            // Add URL Spotter
            var urlSpotter = new PatternSpotter(language, 0, "url", "URL");
            urlSpotter.NewPattern("URL", mp => mp.Add(new PatternUnit(P.Single().LikeURL())));
            _nlp.Add(urlSpotter);

            // Add Phone Spotter (Regex)
            // Simple US Phone regex
            var phoneRegex = @"\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b";
            var phoneSpotter = new RegexSpotter(phoneRegex, "PHONE_NUMBER", language);
            _nlp.Add(phoneSpotter);

            // Add Credit Card Spotter (Regex)
            // Simple regex, not matching all
            var ccRegex = @"\b(?:\d[ -]*?){13,16}\b";
            var ccSpotter = new RegexSpotter(ccRegex, "CREDIT_CARD", language);
            _nlp.Add(ccSpotter);

            // Add IP Address Spotter (Regex)
             var ipRegex = @"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b";
             var ipSpotter = new RegexSpotter(ipRegex, "IP_ADDRESS", language);
             _nlp.Add(ipSpotter);
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
                                             // We consumed up to j for this entity type.
                                             // We don't advance i because overlapping entities might exist.
                                             // But strictly speaking, if we found B...E, we handled this entity instance.
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
