using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    // Exercises the min/max token-length pre-filter that lets the spotters skip hashing a token whose length
    // could not possibly match any stored entry. The tests assert that the optimization does not change any
    // observable recognition result, including at the exact min/max boundaries and across a store/reload cycle.
    public class SpotterLengthOptimizationTests
    {
        private static string[] EntityValues(IDocument doc) =>
            doc.SelectMany(span => span.GetEntities()).Select(e => e.Value).OrderBy(v => v).ToArray();

        [Fact]
        public async Task Spotter_RecognizesEntriesAtLengthBoundaries()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("AI");         // shortest single token (length 2)
            spotter.AddEntry("Curiosity");  // longest single token (length 9)
            spotter.AddEntry("New York");   // multi-gram whose tokens sit inside the window

            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(spotter);

            const string text = "AI at Curiosity is based near New York, not in X or Supercalifragilistic.";

            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            var values = EntityValues(doc);

            Assert.Contains("AI", values);          // exactly at the min length
            Assert.Contains("Curiosity", values);   // exactly at the max length
            Assert.Contains("New York", values);
            Assert.DoesNotContain("X", values);      // below the min length - filtered, must not match
            Assert.DoesNotContain("Supercalifragilistic", values); // above the max length - filtered, must not match
        }

        [Fact]
        public async Task Spotter_RecognitionUnchangedAfterStoreAndReload()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("Curiosity");
            spotter.AddEntry("San Francisco Bay");

            using var ms = new MemoryStream();
            await spotter.StoreAsync(ms);
            ms.Position = 0;

            var reloaded = new Spotter(Language.English, 0, "", "Entity");
            await reloaded.LoadAsync(ms);
            reloaded.TrimExcess();

            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(reloaded);

            const string text = "Curiosity is near the San Francisco Bay area.";
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            var values = EntityValues(doc);

            Assert.Contains("Curiosity", values);
            Assert.Contains("San Francisco Bay", values);
        }

        [Fact]
        public async Task LinkedSpotter_RecognizesEntriesAtLengthBoundaries()
        {
            English.Register();

            var ai        = UID128.New();
            var curiosity = UID128.New();
            var newYork   = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("AI", ai);
            spotter.AddEntry("Curiosity", curiosity);
            spotter.AddEntry("New York", newYork);

            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(spotter);

            const string text = "AI at Curiosity works in New York, not in X.";
            var doc = new Document(text, Language.English);
            nlp.ProcessSingle(doc);
            var values = EntityValues(doc);

            Assert.Contains("AI", values);
            Assert.Contains("Curiosity", values);
            Assert.Contains("New York", values);
            Assert.DoesNotContain("X", values);
        }

        [Fact]
        public async Task PatternSpotter_SetMatchRespectsLengthWindow()
        {
            English.Register();

            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            var spotter = new PatternSpotter(Language.English, 0, "test", "CAP");
            // Set entries all have length 3, so the length pre-filter admits only 3-char tokens for hashing.
            spotter.NewPattern("Animals", mp => mp.Add(new PatternUnit(PatternUnitPrototype.Single().WithTokens(new[] { "cat", "dog" }, ignoreCase: true))));
            nlp.Add(spotter);

            var doc = new Document("Cat dog cats ox", Language.English);
            nlp.ProcessSingle(doc);
            var values = EntityValues(doc);

            Assert.Contains("Cat", values);           // in set (case-insensitive)
            Assert.Contains("dog", values);           // in set
            Assert.DoesNotContain("cats", values);    // length 4, outside the set window
            Assert.DoesNotContain("ox", values);      // length 2, outside the set window
        }
    }
}
