using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestChurch_Slavic_Slavonic_Old_Bulgarian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Church_Slavic_Slavonic_Old_Bulgarian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Church_Slavic_Slavonic_Old_Bulgarian, tagger: false);
            var doc = new Document("Test string", Language.Church_Slavic_Slavonic_Old_Bulgarian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
