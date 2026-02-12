using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestUrdu
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Urdu.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Urdu, tagger: false);
            var doc = new Document("Test string", Language.Urdu);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
