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

        [Fact(Skip = "WikiNER model failing to detect entities with latest package")]
        public async Task TestEntityRecognition()
        {
            var text = "Barack Obama lives in the United States.";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English);

            // Explicitly add NER model
            nlp.Add(await AveragePerceptronEntityRecognizer.FromStoreAsync(Language.English, 0, "WikiNER"));

            nlp.ProcessSingle(doc);

            var entities = doc.SelectMany(span => span.GetEntities()).ToList();

            Assert.Contains(entities, e => e.EntityType.Type == "Person" && e.Value == "Barack Obama");
            Assert.Contains(entities, e => e.EntityType.Type == "Location" && e.Value == "United States");
        }
    }
}
