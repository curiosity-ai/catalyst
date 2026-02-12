using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestBelarusian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Belarusian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Belarusian);
            var doc = new Document("Test string", Language.Belarusian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
