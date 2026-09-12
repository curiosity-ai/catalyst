using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestNorwegian_Bokmal
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Norwegian_Bokmal.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Norwegian_Bokmal, tagger: false);
            var doc = new Document("Test string", Language.Norwegian_Bokmal);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
