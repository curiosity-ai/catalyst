using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestLatvian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Latvian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Latvian, tagger: false);
            var doc = new Document("Test string", Language.Latvian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
