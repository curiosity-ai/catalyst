using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    // Pins how the spotters capture entries made of more than one token: which tokens carry Begin / Inside /
    // End, which entry wins when several could match at the same position, and that a punctuated entry
    // survives tokenization as a single token so it can be matched at all.
    public class SpotterMultiTokenTests
    {
        private static string[] EntityValues(IDocument doc) =>
            doc.SelectMany(span => span.GetEntities()).Select(e => e.Value).OrderBy(v => v).ToArray();

        private static List<(string token, EntityTag tag)> Tagged(IDocument doc, string captureTag) =>
            doc.SelectMany(span => span.Tokens)
               .SelectMany(t => t.EntityTypes.Where(e => e.Type == captureTag).Select(e => (t.Value, e.Tag)))
               .ToList();

        private static async Task<Pipeline> PipelineForAsync() => await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);

        [Fact]
        public async Task Spotter_TagsEveryTokenOfAMultiTokenEntry()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("San Francisco Bay Area");

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("We met in the San Francisco Bay Area last year.", Language.English);
            nlp.ProcessSingle(doc);

            var tagged = Tagged(doc, "Entity");
            Assert.Equal(new[]
            {
                ("San",       EntityTag.Begin),
                ("Francisco", EntityTag.Inside),
                ("Bay",       EntityTag.Inside),
                ("Area",      EntityTag.End),
            }, tagged);

            Assert.Contains("San Francisco Bay Area", EntityValues(doc));
        }

        [Fact]
        public async Task Spotter_PrefersTheLongestEntryStartingAtTheSameToken()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("New York");
            spotter.AddEntry("New York City");

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("Welcome to New York City today.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Contains("New York City", EntityValues(doc));
            Assert.Equal(new[]
            {
                ("New",  EntityTag.Begin),
                ("York", EntityTag.Inside),
                ("City", EntityTag.End),
            }, Tagged(doc, "Entity"));
        }

        [Fact]
        public async Task Spotter_DoesNotMatchAcrossAGap()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("New York");

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("New big York is not a place.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Empty(EntityValues(doc));
        }

        [Fact]
        public async Task LinkedSpotter_MultiTokenEntryCarriesItsUID()
        {
            English.Register();

            var bay = UID128.New();
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("San Francisco Bay", bay);

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("Sailing on the San Francisco Bay.", Language.English);
            nlp.ProcessSingle(doc);

            var entities = doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == "Linked").ToArray();
            Assert.Single(entities);
            Assert.Equal("San Francisco Bay", entities[0].Value);
            Assert.Equal(bay, entities[0].EntityType.TargetUID);
        }

        [Fact]
        public async Task Spotter_HyphenatedEntryIsKeptWholeAndMatched()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "PartNumber");
            spotter.AddEntry("NAS1291-C3M");

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("Replace with NAS1291-C3M before flight.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Contains("NAS1291-C3M", EntityValues(doc));
        }

        [Fact]
        public async Task LinkedSpotter_HyphenatedEntryIsKeptWholeAndMatched()
        {
            English.Register();

            var uid = UID128.New();
            var spotter = new LinkedSpotter(Language.English, 0, "", "PartNumber");
            spotter.AddEntry("NAS1291-C3M", uid);

            var nlp = await PipelineForAsync();
            nlp.Add(spotter);

            var doc = new Document("Replace with NAS1291-C3M before flight.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Contains("NAS1291-C3M", EntityValues(doc));
        }
    }
}
