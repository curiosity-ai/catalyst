using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    public class PoolTrimTests
    {
        [Fact]
        public void RentAfterReturnRecyclesTheSameDocument()
        {
            var pool = new DocumentPool();

            var first = pool.Rent("a document", Language.English);
            pool.Return(first);

            var second = pool.Rent("a document", Language.English);

            Assert.Same(first, second);
        }

        [Fact]
        public void TrimAllReachesAPoolTheCallerMade()
        {
            var pool = new DocumentPool();

            var first = pool.Rent("a document", Language.English);
            pool.Return(first);

            DocumentPool.TrimAll();

            var afterTrim = pool.Rent("a document", Language.English);

            Assert.NotSame(first, afterTrim);
        }

        [Fact]
        public void TrimAllLeavesThePoolUsable()
        {
            var pool = new DocumentPool();

            DocumentPool.TrimAll();

            var document = pool.Rent("a document", Language.English);
            document.AddSpan(0, 9);
            document.ReserveTokens(0, 4);

            Assert.Equal("a document", document.Value);

            pool.Return(document);

            Assert.Same(document, pool.Rent("a document", Language.English));
        }
    }
}
