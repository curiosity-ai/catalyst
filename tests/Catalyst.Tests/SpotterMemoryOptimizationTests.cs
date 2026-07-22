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
        public void Spotter_OnlyRecordsExceptionsForSplittableWordsAndKeepsThemAfterOptimize()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");

            // Words that are all letters and/or digits are not split by the tokenizer, so they need no exception.
            spotter.AddEntry("covid19");
            spotter.AddEntry("New York 2024");
            Assert.Empty(spotter.GetSpecialCases());

            // Words containing punctuation would be split, so they require a "keep as-is" exception.
            spotter.AddEntry("node.js");
            spotter.AddEntry("U.S.A.");
            var before = spotter.GetSpecialCases().Select(kv => kv.Key).OrderBy(k => k).ToArray();
            Assert.Equal(2, before.Length);

            // The same model can be imported into more than one pipeline, so OptimizeMemory must keep the table.
            spotter.OptimizeMemory();
            var after = spotter.GetSpecialCases().Select(kv => kv.Key).OrderBy(k => k).ToArray();
            Assert.Equal(before, after);
        }

        [Fact]
        public void LinkedSpotter_KeepsExceptionsAfterOptimize()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("plain", UID128.New());   // all letters -> no exception
            spotter.AddEntry("node.js", UID128.New()); // punctuation -> needs an exception

            var before = spotter.GetSimpleSpecialCases().OrderBy(k => k).ToArray();
            Assert.Single(before);

            spotter.OptimizeMemory();
            var after = spotter.GetSimpleSpecialCases().OrderBy(k => k).ToArray();
            Assert.Equal(before, after);
        }

        [Fact]
        public void LinkedSpotter_MembershipMode_FreezesReadOnlyAndReportsMemory()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("Curiosity", UID128.New());
            spotter.AddEntry("New York", UID128.New());

            // Providing a resolver switches the freeze to the probabilistic membership tables.
            spotter.SetCaptureResolver(text => default);
            spotter.OptimizeMemory();

            Assert.True(spotter.IsMemoryOptimized);
            Assert.True(spotter.OptimizedMemoryBytes > 0);

            // Membership-only mode is lossy/read-only: mutating (which would need a lossless unfreeze) must throw.
            Assert.Throws<System.InvalidOperationException>(() => spotter.AddEntry("Boston", UID128.New()));
        }

        [Fact]
        public async Task LinkedSpotter_MembershipMode_ResolvesUidAndFiltersFalsePositives()
        {
            English.Register();

            var curiosity = UID128.New();
            var newYork   = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("Curiosity", curiosity);
            spotter.AddEntry("New York", newYork);
            spotter.AddEntry("Ghost", UID128.New()); // a member of the table that the resolver will reject

            // The resolver stands in for the graph: it accepts only real entities and supplies their UID.
            spotter.SetCaptureResolver(text =>
            {
                var s = text.ToString();
                if (s == "Curiosity") { return curiosity; }
                if (s == "New York") { return newYork; }
                return default; // not in the graph -> drop (covers "Ghost" and any fingerprint false positive)
            });

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            spotter.OptimizeMemory();
            Assert.True(spotter.IsMemoryOptimized);

            var doc = new Document("Curiosity works in New York with Ghost.", Language.English);
            nlp.ProcessSingle(doc);

            var values = EntityValues(doc);
            Assert.Contains("Curiosity", values);
            Assert.Contains("New York", values);
            Assert.DoesNotContain("Ghost", values); // table-member, but the resolver rejected it

            // the emitted UID is the one the resolver returned
            var entities = doc.SelectMany(s => s.GetEntities()).ToArray();
            Assert.Contains(entities, e => e.Value == "New York" && e.EntityType.TargetUID == newYork);
            Assert.Contains(entities, e => e.Value == "Curiosity" && e.EntityType.TargetUID == curiosity);
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
