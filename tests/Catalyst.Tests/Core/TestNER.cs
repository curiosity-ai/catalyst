using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using System.Linq;

namespace Catalyst.Tests.Core
{
    public class TestNER
    {
        public TestNER()
        {
            Catalyst.Models.English.Register();
        }

        [Fact(Skip = "Known issue with MessagePack serialization for Tagger/NER models")]
        public async Task TestEntityRecognition()
        {
            var text = "John Smith lives in New York and works for Microsoft.";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English); // Requires tagger
            nlp.ProcessSingle(doc);

            var entities = doc.SelectMany(span => span.GetEntities()).ToList();

            Assert.Contains(entities, e => e.EntityType.Type == "Person" && e.Value == "John Smith");
            Assert.Contains(entities, e => e.EntityType.Type == "Location" && e.Value == "New York");
            Assert.Contains(entities, e => e.EntityType.Type == "Organization" && e.Value == "Microsoft");
        }
    }
}
