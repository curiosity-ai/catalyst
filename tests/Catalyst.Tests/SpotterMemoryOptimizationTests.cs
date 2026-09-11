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

            // Compaction happens the first time the model is used - joining a pipeline already needs its
            // tokenizer-exception table - so "not yet optimized" is the state right after training it.
            Assert.False(spotter.IsMemoryOptimized);

            var nlp = await Pipeline.ForAsync(Language.English);
            nlp.Add(spotter);

            const string text = "Curiosity is based near New York and the San Francisco Bay area.";

            var before = new Document(text, Language.English);
            nlp.ProcessSingle(before);
            var beforeValues = EntityValues(before);

            Assert.True(spotter.IsMemoryOptimized);
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
        public void Spotter_OnlyRecordsExceptionsForWordsTheTokenizerWouldSplit()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");

            // Words the tokenizer already keeps whole need no exception - and that covers far more than
            // "all letters and digits": a hyphen, a slash or a dot between alphanumerics never splits.
            spotter.AddEntry("covid19");
            spotter.AddEntry("New York 2024");
            spotter.AddEntry("node.js");
            spotter.AddEntry("NAS1291-C3M");
            spotter.AddEntry("D38999/24WC35PN");
            Assert.Empty(spotter.GetSimpleSpecialCases().Hashes());

            // Words the tokenizer really would break apart still need a "keep as-is" exception.
            spotter.AddEntry("AT&T");
            spotter.AddEntry("fish,chips");
            var before = spotter.GetSimpleSpecialCases().Hashes().OrderBy(k => k).ToArray();
            Assert.Equal(2, before.Length);

            // The same model can be imported into more than one pipeline, so OptimizeMemory must keep the table.
            spotter.OptimizeMemory();
            var after = spotter.GetSimpleSpecialCases().Hashes().OrderBy(k => k).ToArray();
            Assert.Equal(before, after);
        }

        [Fact]
        public void LinkedSpotter_KeepsExceptionsAfterOptimize()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("plain", UID128.New());    // kept whole by the tokenizer -> no exception
            spotter.AddEntry("node.js", UID128.New());  // also kept whole -> no exception
            spotter.AddEntry("AT&T", UID128.New());     // really would be split -> needs an exception

            var before = spotter.GetSimpleSpecialCases().Hashes().OrderBy(k => k).ToArray();
            Assert.Single(before);

            spotter.OptimizeMemory();
            var after = spotter.GetSimpleSpecialCases().Hashes().OrderBy(k => k).ToArray();
            Assert.Equal(before, after);
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
