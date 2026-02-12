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

            // Add Phone Spotter
            var phoneSpotter = new PatternSpotter(language, 0, "phone", "PHONE_NUMBER");
            // Matches 555-123-4567 (Single token)
            phoneSpotter.NewPattern("Phone-US-Single", mp => mp.Add(new PatternUnit(P.Single().WithShape("999-999-9999"))));
            // Matches (555) 123-4567
            phoneSpotter.NewPattern("Phone-US-Parens", mp => mp.Add(
                new PatternUnit(P.Single().IsOpeningParenthesis()),
                new PatternUnit(P.Single().IsNumeric().WithLength(3, 3)),
                new PatternUnit(P.Single().IsClosingParenthesis()),
                new PatternUnit(P.Single().WithShape("999-9999"))
            ));
            _nlp.Add(phoneSpotter);

            // Add Credit Card Spotter
            var ccSpotter = new PatternSpotter(language, 0, "cc", "CREDIT_CARD");
            // Matches 1234 5678 1234 5678
            ccSpotter.NewPattern("CC-4x4", mp => mp.Add(
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4)),
                new PatternUnit(P.Single().IsNumeric().WithLength(4, 4))
            ));
            _nlp.Add(ccSpotter);

            // Add IP Address Spotter
             var ipSpotter = new PatternSpotter(language, 0, "ip", "IP_ADDRESS");
             // Matches 192.168.1.1
             ipSpotter.NewPattern("IP-v4", mp => mp.Add(new PatternUnit(P.Single().WithShape("999.999.9.9"))));
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
