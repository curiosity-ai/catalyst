using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestIrish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Irish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Irish);
            var doc = new Document("Test string", Language.Irish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
