using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestMarathi
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Marathi.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Marathi, tagger: false);
            var doc = new Document("Test string", Language.Marathi);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
