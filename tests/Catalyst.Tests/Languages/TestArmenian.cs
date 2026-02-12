using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestArmenian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Armenian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Armenian, tagger: false);
            var doc = new Document("Test string", Language.Armenian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
