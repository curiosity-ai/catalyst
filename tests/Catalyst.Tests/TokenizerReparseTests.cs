using System.Linq;
using Catalyst.Models;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    /// <summary>
    /// Tokenizing a span adds to what it already holds, which is what the sentence detector relies on. A
    /// document that was tokenized before is the case where that is wrong, and it is not hypothetical: an
    /// index that re-parses stored documents doubled every one of them on every pass.
    /// </summary>
    public class TokenizerReparseTests
    {
        private const string TEXT = "Curiosity GmbH is based in Berlin. The quick brown fox jumps over the lazy dog. "
                                  + "Fuselage section 13 forward skin panels were re-sealed on 2026-07-14.";

        private static Document Parsed(FastTokenizer tokenizer, int times)
        {
            var document = new Document(TEXT, Language.English);

            for (int i = 0; i < times; i++)
            {
                tokenizer.Parse(document);
            }

            return document;
        }

        [Fact]
        public void ParsingADocumentTwiceLeavesTheSameTokensAsParsingItOnce()
        {
            var tokenizer = new FastTokenizer(Language.English);

            var once  = Parsed(tokenizer, 1);
            var twice = Parsed(tokenizer, 2);

            Assert.Equal(once.TokensCount, twice.TokensCount);
            Assert.Equal(once.SpansCount,  twice.SpansCount);
            Assert.Equal(once.Spans.SelectMany(s => s.Tokens).Select(t => t.Value),
                         twice.Spans.SelectMany(s => s.Tokens).Select(t => t.Value));
        }

        [Fact]
        public void ParsingStaysStableHoweverManyTimesItIsRepeated()
        {
            var tokenizer = new FastTokenizer(Language.English);
            var expected  = Parsed(tokenizer, 1).TokensCount;

            Assert.True(expected > 0);

            for (int times = 2; times <= 5; times++)
            {
                Assert.Equal(expected, Parsed(tokenizer, times).TokensCount);
            }
        }

        [Fact]
        public void APooledDocumentReparsesTheSameWay()
        {
            //The pooled document is what the indexes actually hand the tokenizer, and its Clear returns the
            //span's collections to the pool - so re-parsing has to leave it usable, not just correctly sized.
            var tokenizer = new FastTokenizer(Language.English);
            var pool      = new DocumentPool();
            var document  = pool.Rent(TEXT, Language.English);

            tokenizer.Parse(document);

            var expected = document.TokensCount;

            tokenizer.Parse(document);
            tokenizer.Parse(document);

            Assert.Equal(expected, document.TokensCount);
            Assert.True(document.Spans.All(s => s.TokensCount > 0));

            pool.Return(document);
        }

        [Fact]
        public void ASpanSetUpWithoutTokensIsStillTokenizedIntoRatherThanReplaced()
        {
            //Only a document that already has tokens is reset: one whose spans were laid out by something else
            //has to keep them, which is how the sentence detector and the tokenizer work together.
            var tokenizer = new FastTokenizer(Language.English);
            var document  = new Document(TEXT, Language.English);

            document.AddSpan(0, 33);
            document.AddSpan(34, TEXT.Length - 1);

            tokenizer.Parse(document);

            Assert.Equal(2, document.SpansCount);
            Assert.True(document.TokensCount > 0);
        }
    }
}
