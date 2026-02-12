using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestSpanish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Spanish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Spanish, tagger: false);
            var doc = new Document("Test string", Language.Spanish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
