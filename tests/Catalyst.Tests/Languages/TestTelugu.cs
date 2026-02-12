using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestTelugu
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Telugu.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Telugu, tagger: false);
            var doc = new Document("Test string", Language.Telugu);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
