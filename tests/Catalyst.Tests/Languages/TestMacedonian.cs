using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestMacedonian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Macedonian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Macedonian, tagger: false);
            var doc = new Document("Test string", Language.Macedonian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
