using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestMaltese
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Maltese.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Maltese, tagger: false);
            var doc = new Document("Test string", Language.Maltese);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
