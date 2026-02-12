using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestRussian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Russian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Russian);
            var doc = new Document("Test string", Language.Russian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
