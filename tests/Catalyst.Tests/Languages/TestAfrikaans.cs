using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestAfrikaans
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Afrikaans.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Afrikaans);
            var doc = new Document("Test string", Language.Afrikaans);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
