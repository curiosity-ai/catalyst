using System.Collections.Generic;
using System.Linq;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    /// <summary>
    /// <see cref="Span.GetEntities"/> and <see cref="Span.GetCapturedTokens"/> read a token at a time and
    /// used to do it through the boxing indexer and an <c>OrderBy</c> per token. These pin the shapes they
    /// hand back, which is what the two-pass rewrite has to keep: every Begin is tried before any Single,
    /// the longest match wins, and an unterminated Begin captures nothing.
    /// </summary>
    public class SpanEntityReadingTests
    {
        private const string TEXT = "aaa bbb ccc ddd eee fff ggg hhh";

        private static Span BuildSpan(params (int token, string type, EntityTag tag)[] entities)
        {
            var document = new Document(TEXT, Language.English);
            var span     = document.AddSpan(0, TEXT.Length - 1);

            for (int i = 0; i < 8; i++)
            {
                span.AddTokenAsStruct(i * 4, i * 4 + 2);
            }

            foreach (var (token, type, tag) in entities)
            {
                span.GetTokenAsStruct(token).AddEntityType(new EntityType(type, tag));
            }

            return span;
        }

        private static List<(string Value, string Type)> Read(IEnumerable<ITokens> tokens)
        {
            return tokens.Select(t => (t.Value, t.EntityType.Type)).ToList();
        }

        [Fact]
        public void ASingleEntityIsOneCapturedToken()
        {
            var read = Read(BuildSpan((2, "Part", EntityTag.Single)).GetEntities());

            Assert.Equal(new[] { ("ccc", "Part") }, read);
        }

        [Fact]
        public void EveryTokenOfAMultiTokenEntityIsCaptured()
        {
            var read = Read(BuildSpan((1, "Part", EntityTag.Begin),
                                      (2, "Part", EntityTag.Inside),
                                      (3, "Part", EntityTag.End)).GetEntities());

            Assert.Equal(new[] { ("bbb ccc ddd", "Part") }, read);
        }

        [Fact]
        public void TheLongestOfTwoEntitiesStartingOnTheSameTokenWins()
        {
            var read = Read(BuildSpan((0, "Short", EntityTag.Begin),
                                      (0, "Long",  EntityTag.Begin),
                                      (1, "Short", EntityTag.End),
                                      (1, "Long",  EntityTag.Inside),
                                      (2, "Long",  EntityTag.End)).GetEntities());

            Assert.Equal(new[] { ("aaa bbb ccc", "Long") }, read);
        }

        [Fact]
        public void ABeginThatNeverEndsCapturesNothing()
        {
            Assert.Empty(BuildSpan((0, "Part", EntityTag.Begin), (1, "Part", EntityTag.Inside)).GetEntities());
        }

        [Fact]
        public void ABeginIsPreferredOverASingleOnTheSameToken()
        {
            var read = Read(BuildSpan((0, "Alone",  EntityTag.Single),
                                      (0, "Joined", EntityTag.Begin),
                                      (1, "Joined", EntityTag.End)).GetEntities());

            Assert.Equal(new[] { ("aaa bbb", "Joined") }, read);
        }

        [Fact]
        public void EverySingleOnATokenIsReturnedWhenNoBeginMatches()
        {
            var read = Read(BuildSpan((0, "First",  EntityTag.Single),
                                      (0, "Second", EntityTag.Single)).GetEntities());

            Assert.Equal(new[] { ("aaa", "First"), ("aaa", "Second") }, read);
        }

        [Fact]
        public void TheFilterDecidesWhichEntitiesAreRead()
        {
            var span = BuildSpan((0, "Keep", EntityTag.Single), (2, "Drop", EntityTag.Single));

            Assert.Equal(new[] { ("aaa", "Keep") }, Read(span.GetEntities(et => et.Type == "Keep")));
        }

        [Fact]
        public void GetCapturedTokensReturnsEveryTokenWithEntitiesFolded()
        {
            var read = BuildSpan((1, "Part", EntityTag.Begin),
                                 (2, "Part", EntityTag.End),
                                 (5, "Other", EntityTag.Single)).GetCapturedTokens().Select(t => t.Value).ToList();

            Assert.Equal(new[] { "aaa", "bbb ccc", "ddd", "eee", "fff", "ggg", "hhh" }, read);
        }

        [Fact]
        public void GetTokenAsStructReadsTheSameTokenAsTheIndexer()
        {
            var span = BuildSpan((3, "Part", EntityTag.Single));

            for (int i = 0; i < span.TokensCount; i++)
            {
                var boxed = span[i];
                var plain = span.GetTokenAsStruct(i);

                Assert.Equal(boxed.Begin,                plain.Begin);
                Assert.Equal(boxed.End,                  plain.End);
                Assert.Equal(boxed.Value,                plain.Value);
                Assert.Equal(boxed.Index,                plain.Index);
                Assert.Equal(boxed.EntityTypes.Count,    plain.EntityTypes.Count);
            }
        }
    }
}
