using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestHebrew
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Hebrew.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Hebrew, tagger: false);
            var doc = new Document("Test string", Language.Hebrew);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
