using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestCroatian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Croatian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Croatian, tagger: false);
            var doc = new Document("Test string", Language.Croatian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
