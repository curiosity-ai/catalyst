using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestArabic
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Arabic.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Arabic, tagger: false);
            var doc = new Document("Test string", Language.Arabic);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
