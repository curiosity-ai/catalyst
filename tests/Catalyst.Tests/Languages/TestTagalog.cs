using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestTagalog
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Tagalog.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Tagalog, tagger: false);
            var doc = new Document("Test string", Language.Tagalog);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
