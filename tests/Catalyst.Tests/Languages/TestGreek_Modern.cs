using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestGreek_Modern
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Greek_Modern.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Greek_Modern, tagger: false);
            var doc = new Document("Test string", Language.Greek_Modern);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
