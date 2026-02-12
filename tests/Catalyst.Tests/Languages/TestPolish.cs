using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestPolish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Polish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Polish, tagger: false);
            var doc = new Document("Test string", Language.Polish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
