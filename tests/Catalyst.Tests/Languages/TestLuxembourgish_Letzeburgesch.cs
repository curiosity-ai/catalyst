using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestLuxembourgish_Letzeburgesch
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Luxembourgish_Letzeburgesch.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Luxembourgish_Letzeburgesch, tagger: false);
            var doc = new Document("Test string", Language.Luxembourgish_Letzeburgesch);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
