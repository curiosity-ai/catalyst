using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestIcelandic
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Icelandic.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Icelandic);
            var doc = new Document("Test string", Language.Icelandic);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
