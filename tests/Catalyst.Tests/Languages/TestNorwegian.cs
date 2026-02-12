using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestNorwegian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Norwegian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Norwegian);
            var doc = new Document("Test string", Language.Norwegian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
