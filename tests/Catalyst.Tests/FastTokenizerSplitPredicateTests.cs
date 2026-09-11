using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    // FastTokenizer.WouldSplit decides whether a spotter has to register a tokenization exception for a word.
    // It is a reimplementation of the decisions the tokenizer takes on a whitespace-delimited candidate, so
    // these tests pin it against what the tokenizer actually does - if the two ever disagree, a model either
    // stores exceptions it does not need (memory) or misses one it does (a word silently stops matching).
    public class FastTokenizerSplitPredicateTests
    {
        private static async Task<Pipeline> PlainPipelineAsync()
        {
            English.Register();
            return await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
        }

        // Tokenizes "x <word> y" and reports whether <word> survived as exactly one token.
        private static bool TokenizerSplits(Pipeline nlp, string word)
        {
            var doc = new Document("x " + word + " y", Language.English);
            nlp.ProcessSingle(doc);
            var tokens = doc.SelectMany(s => s.Tokens).Select(t => t.Value).ToArray();
            return !(tokens.Length == 3 && tokens[0] == "x" && tokens[1] == word && tokens[2] == "y");
        }

        [Theory]
        // Shapes a part-number or product-code catalogue is full of - all kept whole, so none of them needs
        // an exception even though none is "all letters or digits".
        [InlineData("NAS1291-C3M")]
        [InlineData("D38999/24WC35PN")]
        [InlineData("M83248/1-012")]
        [InlineData("7075-T6")]
        [InlineData("node.js")]
        [InlineData("co-op")]
        [InlineData("3.5mm")]
        [InlineData("x/y/z")]
        [InlineData("50%")]
        [InlineData("C++")]
        [InlineData("a+b")]
        [InlineData("a@b")]
        // Shapes that really are broken apart.
        [InlineData("AT&T")]
        [InlineData("fish,chips")]
        [InlineData("one_two")]
        [InlineData("12:30")]
        [InlineData("a;b")]
        [InlineData("a!b")]
        [InlineData("a?b")]
        [InlineData("a'b")]
        [InlineData("a*b")]
        [InlineData("a..b")]
        [InlineData("a--b")]
        [InlineData("e.Mail")]
        [InlineData("#tag")]
        [InlineData("(NAS1291)")]
        [InlineData("R$5")]
        public async Task PredicateAgreesWithTheTokenizer(string word)
        {
            var nlp = await PlainPipelineAsync();
            Assert.Equal(TokenizerSplits(nlp, word), FastTokenizer.WouldSplit(word.AsSpan(), Language.English));
        }

        [Fact]
        public async Task PredicateAgreesWithTheTokenizerOverGeneratedShapes()
        {
            var nlp = await PlainPipelineAsync();

            const string alphanumeric = "abzABZ019";
            const string punctuation  = "-/._,;:!?'\"()[]&$%#*+=@|~^<>{}\\";
            var rng = new Random(20260911);

            var disagreements = new List<string>();
            var seen          = new HashSet<string>();
            var builder       = new StringBuilder();

            for (int i = 0; i < 4000; i++)
            {
                builder.Clear();
                int length = rng.Next(2, 9);

                for (int j = 0; j < length; j++)
                {
                    // Mostly alphanumeric with punctuation sprinkled in, which is what real catalogue values
                    // look like and keeps the generated shapes in the region that actually matters.
                    builder.Append(rng.NextDouble() < 0.65 ? alphanumeric[rng.Next(alphanumeric.Length)]
                                                           : punctuation[rng.Next(punctuation.Length)]);
                }

                var word = builder.ToString();
                if (word.Any(char.IsWhiteSpace) || !seen.Add(word)) { continue; }

                if (TokenizerSplits(nlp, word) != FastTokenizer.WouldSplit(word.AsSpan(), Language.English))
                {
                    disagreements.Add(word);
                }
            }

            Assert.True(disagreements.Count == 0, $"WouldSplit disagreed with the tokenizer on {disagreements.Count} of {seen.Count} shapes, e.g. {string.Join(", ", disagreements.Take(10))}");
        }

        [Fact]
        public async Task AWordNeedingAnExceptionStillMatchesEndToEnd()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("AT&T");

            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(spotter);

            var doc = new Document("She works at AT&T today.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Contains("AT&T", doc.SelectMany(s => s.GetEntities()).Select(e => e.Value));
        }
    }
}
