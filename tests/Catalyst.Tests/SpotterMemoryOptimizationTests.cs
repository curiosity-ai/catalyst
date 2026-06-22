using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    public class SpotterMemoryOptimizationTests
    {
        private static string[] EntityValues(IDocument doc) =>
            doc.SelectMany(span => span.GetEntities()).Select(e => e.Value).OrderBy(v => v).ToArray();

        [Fact]
        public async Task Spotter_FrozenRecognitionMatchesUnfrozen()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("Curiosity");
            spotter.AddEntry("New York");
            spotter.AddEntry("San Francisco Bay");

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            const string text = "Curiosity is based near New York and the San Francisco Bay area.";

            var before = new Document(text, Language.English);
            nlp.ProcessSingle(before);
            var beforeValues = EntityValues(before);

            Assert.False(spotter.IsMemoryOptimized);
            Assert.Contains("Curiosity", beforeValues);
            Assert.Contains("New York", beforeValues);
            Assert.Contains("San Francisco Bay", beforeValues);

            spotter.TrimExcess(); // triggers the compaction (freeze)
            Assert.True(spotter.IsMemoryOptimized);
            Assert.True(spotter.OptimizedMemoryBytes > 0);

            var after = new Document(text, Language.English);
            nlp.ProcessSingle(after);

            Assert.Equal(beforeValues, EntityValues(after));
        }

        [Fact]
        public async Task LinkedSpotter_FrozenRecognitionMatchesUnfrozen()
        {
            English.Register();

            var curiosity = UID128.New();
            var newYork   = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("Curiosity", curiosity);
            spotter.AddEntry("New York", newYork);

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            const string text = "Curiosity works in New York.";

            var before = new Document(text, Language.English);
            nlp.ProcessSingle(before);
            var beforeValues = EntityValues(before);

            spotter.TrimExcess();
            Assert.True(spotter.IsMemoryOptimized);

            var after = new Document(text, Language.English);
            nlp.ProcessSingle(after);

            Assert.Equal(beforeValues, EntityValues(after));
            Assert.Contains("Curiosity", beforeValues);
            Assert.Contains("New York", beforeValues);
        }

        [Fact]
        public async Task Spotter_FingerprintModeStillRecognizes()
        {
            English.Register();

            var previous = SpotterCompaction.UseFingerprint32;
            SpotterCompaction.UseFingerprint32 = true;
            try
            {
                var spotter = new Spotter(Language.English, 0, "", "Entity");
                spotter.AddEntry("Curiosity");
                spotter.AddEntry("New York");

                var nlp = await Pipeline.ForAsync(Language.English);
                nlp.Add(spotter);

                spotter.TrimExcess(); // freezes using fingerprint compression
                Assert.True(spotter.IsMemoryOptimized);

                var doc = new Document("Curiosity is in New York.", Language.English);
                nlp.ProcessSingle(doc);

                var values = EntityValues(doc);
                Assert.Contains("Curiosity", values);
                Assert.Contains("New York", values);
            }
            finally
            {
                SpotterCompaction.UseFingerprint32 = previous;
            }
        }

        [Fact]
        public async Task Spotter_CanMutateAfterFreezeInExactMode()
        {
            English.Register();

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("Curiosity");

            spotter.TrimExcess();
            Assert.True(spotter.IsMemoryOptimized);

            // Adding a new entry after freezing must transparently rehydrate and apply.
            spotter.AddEntry("New York");
            Assert.False(spotter.IsMemoryOptimized);

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            var doc = new Document("Curiosity is in New York.", Language.English);
            nlp.ProcessSingle(doc);

            var values = EntityValues(doc);
            Assert.Contains("Curiosity", values);
            Assert.Contains("New York", values);
        }
    }
}
