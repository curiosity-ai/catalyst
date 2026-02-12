using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestVietnamese
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Vietnamese.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Vietnamese);
            var doc = new Document("Test string", Language.Vietnamese);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
