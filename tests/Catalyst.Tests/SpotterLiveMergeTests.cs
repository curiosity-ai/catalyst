using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    // A spotter keeps its entries in segments: what is added after a flush becomes a small segment beside the
    // ones already built, and segments are merged once they are of comparable size. These tests pin what that
    // has to preserve - the newest segment decides an entry, a removal hides an older copy, a removal never
    // takes a name away from another node, multi-token entries span segments - and what it is for: adding a
    // few entries to a large model neither rebuilds nor reallocates it.
    public class SpotterLiveMergeTests
    {
        private static async Task<Pipeline> PipelineWithAsync(IProcess process)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(process);
            return nlp;
        }

        private static Dictionary<string, UID128> Captured(IDocument doc, string captureTag) =>
            doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == captureTag).GroupBy(e => e.Value).ToDictionary(g => g.Key, g => g.Last().EntityType.TargetUID);

        private static LinkedSpotter LargeModel(int entries, out Dictionary<string, UID128> expected, params (string name, UID128 uid)[] extra)
        {
            expected    = new Dictionary<string, UID128>();
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");

            foreach (var (name, uid) in extra)
            {
                spotter.AddEntry(name, uid);
                expected[name] = uid;
            }

            for (int i = 0; i < entries; i++)
            {
                var name = "PN" + i.ToString("000000");
                var uid  = UID128.New();
                spotter.AddEntry(name, uid);
                expected[name] = uid;
            }

            spotter.Flush();
            return spotter;
        }

        [Fact]
        public async Task LinkedSpotter_AddsAfterFlushAsANewSegmentAndMatchesBoth()
        {
            var spotter = LargeModel(5_000, out var expected);
            Assert.Equal(1, spotter.SegmentCount);

            var added = UID128.New();
            spotter.AddEntry("Curiosity", added);
            spotter.Flush();

            Assert.Equal(2, spotter.SegmentCount);

            var doc = new Document("PN000042 was shipped by Curiosity.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var captured = Captured(doc, "Linked");
            Assert.Equal(expected["PN000042"], captured["PN000042"]);
            Assert.Equal(added,                captured["Curiosity"]);
        }

        [Fact]
        public void LinkedSpotter_ANewerSegmentReplacesTheValueOfAnOlderOne()
        {
            var spotter = LargeModel(5_000, out var expected);

            var replacement = UID128.New();
            spotter.AddEntry("PN000007", replacement);
            spotter.Flush();

            Assert.True(spotter.TryGetValue("PN000007", out var value));
            Assert.Equal(replacement, value);
            Assert.True(spotter.TryGetValue("PN000008", out value));
            Assert.Equal(expected["PN000008"], value);
        }

        [Fact]
        public async Task LinkedSpotter_AnEditedNameStopsMatchingItsOldForm()
        {
            var spotter = LargeModel(5_000, out var expected);
            var node    = expected["PN000100"];

            // A node renamed from PN000100 to PN-RENAMED: the old name goes, the new one comes.
            spotter.RemoveEntry("PN000100", node);
            spotter.AddEntry("PN-RENAMED", node);
            spotter.Flush();

            var doc = new Document("Replace PN000100 with PN-RENAMED and keep PN000101.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var captured = Captured(doc, "Linked");
            Assert.False(captured.ContainsKey("PN000100"));
            Assert.Equal(node,                 captured["PN-RENAMED"]);
            Assert.Equal(expected["PN000101"], captured["PN000101"]);
            Assert.False(spotter.TryGetValue("PN000100", out _));
        }

        [Fact]
        public void LinkedSpotter_ARemovalLeavesANameAnotherNodeHolds()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            var first   = UID128.New();
            var second  = UID128.New();

            spotter.AddEntry("Acme", first);
            spotter.Flush();

            // A second node takes the same name; then the first one lets go of it.
            spotter.AddEntry("Acme", second);
            spotter.Flush();
            spotter.RemoveEntry("Acme", first);
            spotter.Flush();

            Assert.True(spotter.TryGetValue("Acme", out var value));
            Assert.Equal(second, value);
        }

        [Fact]
        public void LinkedSpotter_ResolvesAddsAndRemovalsInOrderWithinOneFlush()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            var node    = UID128.New();
            var other   = UID128.New();

            spotter.AddEntry("Kept", node);
            spotter.AddEntry("Gone", node);
            spotter.Flush();

            spotter.RemoveEntry("Gone", node);
            spotter.AddEntry("Gone", other);      // re-added afterwards, by someone else
            spotter.AddEntry("Fresh", node);
            spotter.RemoveEntry("Fresh", other);  // not linked to other: ignored
            spotter.RemoveEntry("Kept", other);   // not linked to other: ignored
            spotter.Flush();

            var entries = spotter.GetEntries().ToDictionary(kv => kv.Key, kv => kv.Value);
            Assert.Equal(new[] { "Fresh", "Gone", "Kept" }, entries.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
            Assert.Equal(other, entries["Gone"]);
            Assert.Equal(node,  entries["Fresh"]);
            Assert.Equal(node,  entries["Kept"]);
        }

        [Fact]
        public async Task LinkedSpotter_MultiTokenEntriesSpanSegments()
        {
            var newYork = UID128.New();
            var city    = UID128.New();
            var spotter = LargeModel(5_000, out _, ("New York", newYork));

            spotter.AddEntry("New York City", city);
            spotter.Flush();
            Assert.Equal(2, spotter.SegmentCount);

            var nlp = await PipelineWithAsync(spotter);

            var doc = new Document("I live in New York City now.", Language.English);
            nlp.ProcessSingle(doc);
            Assert.Equal(city, Captured(doc, "Linked")["New York City"]);

            // Removing the longer entry falls back to the shorter one held by the older segment.
            spotter.RemoveEntry("New York City", city);
            spotter.Flush();

            doc = new Document("I live in New York City now.", Language.English);
            nlp.ProcessSingle(doc);
            var captured = Captured(doc, "Linked");
            Assert.False(captured.ContainsKey("New York City"));
            Assert.Equal(newYork, captured["New York"]);
        }

        [Fact]
        public async Task LinkedSpotter_AWordTheTokenizerWouldSplitMatchesOnceAddedToARunningPipeline()
        {
            var spotter = LargeModel(2_000, out _);
            var nlp     = await PipelineWithAsync(spotter);

            var att = UID128.New();
            spotter.AddEntry("AT&T", att);
            spotter.Flush();

            var doc = new Document("Call AT&T today.", Language.English);
            nlp.ProcessSingle(doc);

            Assert.Equal(att, Captured(doc, "Linked")["AT&T"]);
        }

        [Fact]
        public void LinkedSpotter_KeepsEveryTokenizerExceptionAcrossManySmallFlushes()
        {
            var spotter = LargeModel(2_000, out _);
            var words   = Enumerable.Range(0, 6_000).Select(i => "R&D" + i).ToArray();

            // Past the 4,096 the exception table keeps beside itself, the additions are folded into the table.
            for (int i = 0; i < words.Length; i++)
            {
                spotter.AddEntry(words[i], UID128.New());
                if (i % 100 == 99) { spotter.Flush(); }
            }
            spotter.Flush();

            var exceptions = spotter.GetSimpleSpecialCases();
            Assert.Equal(words.Length, exceptions.Count);
            Assert.All(words, w => Assert.True(exceptions.Contains(w.CaseSensitiveHash32())));
        }

        [Fact]
        public void LinkedSpotter_KeepsTheSegmentCountLogarithmicAndAgreesWithABulkBuild()
        {
            var incremental = new LinkedSpotter(Language.English, 0, "", "Linked");
            var expected    = new Dictionary<string, UID128>();
            int maxSegments = 0;

            for (int i = 0; i < 40_000; i++)
            {
                var name = "E" + i.ToString("000000");
                var uid  = UID128.New();
                incremental.AddEntry(name, uid);
                expected[name] = uid;

                if (i % 37 == 0)
                {
                    incremental.Flush();
                    maxSegments = Math.Max(maxSegments, incremental.SegmentCount);
                }
            }

            incremental.Flush();

            // Sizes fall by MERGE_FACTOR (4) from one segment to the next, starting from a 1,024-entry floor.
            Assert.InRange(maxSegments, 1, 2 + (int)Math.Ceiling(Math.Log(40_000 / 1024.0, 4)));

            var entries = incremental.GetEntries().ToList();
            Assert.Equal(expected.Count, entries.Count);
            foreach (var (key, value) in entries) { Assert.Equal(expected[key], value); }
        }

        [Fact]
        public void LinkedSpotter_AgreesWithADictionaryOverRandomEdits()
        {
            var random  = new Random(1234);
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            var oracle  = new Dictionary<string, UID128>();
            var nodes   = Enumerable.Range(0, 50).Select(_ => UID128.New()).ToArray();

            for (int step = 0; step < 20_000; step++)
            {
                var name = "N" + random.Next(3_000).ToString("0000");
                var node = nodes[random.Next(nodes.Length)];

                if (random.Next(3) == 0)
                {
                    spotter.RemoveEntry(name, node);
                    if (oracle.TryGetValue(name, out var linked) && linked == node) { oracle.Remove(name); }
                }
                else
                {
                    spotter.AddEntry(name, node);
                    oracle[name] = node;
                }

                if (random.Next(50) == 0) { spotter.Flush(); }
            }

            spotter.Flush();

            var entries = spotter.GetEntries().ToDictionary(kv => kv.Key, kv => kv.Value);
            Assert.Equal(oracle.Count, entries.Count);
            foreach (var (key, value) in oracle) { Assert.Equal(value, entries[key]); }

            for (int i = 0; i < 3_000; i++)
            {
                var name = "N" + i.ToString("0000");
                Assert.Equal(oracle.TryGetValue(name, out var want), spotter.TryGetValue(name, out var got));
                Assert.Equal(want, got);
            }
        }

        [Fact]
        public async Task LinkedSpotter_StoresOneSegmentWithoutRemovalsAndReloads()
        {
            var spotter = LargeModel(3_000, out var expected);
            spotter.RemoveEntry("PN000001", expected["PN000001"]);
            spotter.AddEntry("Curiosity", UID128.New());
            spotter.Flush();
            Assert.True(spotter.SegmentCount > 1);

            using var stream = new MemoryStream();
            await spotter.StoreAsync(stream);
            Assert.Equal(1, spotter.SegmentCount);

            stream.Seek(0, SeekOrigin.Begin);
            var reloaded = new LinkedSpotter(Language.English, 0, "", "Linked");
            await reloaded.LoadAsync(stream);
            reloaded.TrimExcess();

            Assert.False(reloaded.TryGetValue("PN000001", out _));
            Assert.True(reloaded.TryGetValue("PN000002", out var value));
            Assert.Equal(expected["PN000002"], value);
            Assert.True(reloaded.TryGetValue("Curiosity", out _));
            Assert.Equal(expected.Count, reloaded.GetEntries().Count());
        }

        [Fact]
        public void LinkedSpotter_AddingAFewEntriesDoesNotReallocateALargeModel()
        {
            var spotter = LargeModel(200_000, out _);
            long modelBytes = spotter.OptimizedMemoryBytes;

            // Warm up the code paths so the measurement sees the steady state, not JIT or pool first-use.
            spotter.AddEntry("warm-up", UID128.New());
            spotter.Flush();

            long before = GC.GetAllocatedBytesForCurrentThread();

            for (int i = 0; i < 10; i++) { spotter.AddEntry("Fresh" + i, UID128.New()); }
            spotter.Flush();

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

            // The small segments are merged with each other, never with the 200,000-entry one.
            Assert.True(allocated < modelBytes / 10, $"Adding 10 entries allocated {allocated:n0} bytes against a {modelBytes:n0}-byte model.");
            Assert.True(spotter.TryGetValue("Fresh3", out _));
        }

        [Fact]
        public async Task LinkedSpotter_OnceFlushedExplicitlyOnlyTheOwnerPublishesABatch()
        {
            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            var node    = UID128.New();

            spotter.AddEntry("Warp Coil", node);
            spotter.Flush();

            var nlp = await PipelineWithAsync(spotter);

            // Half of a rename: recognition must not publish it on the writer's behalf.
            spotter.RemoveEntry("Warp Coil", node);

            var doc = new Document("Check the Warp Coil.", Language.English);
            nlp.ProcessSingle(doc);
            Assert.Equal(node, Captured(doc, "Linked")["Warp Coil"]);

            spotter.AddEntry("Warp Core", node);
            spotter.Flush();

            doc = new Document("Check the Warp Coil and the Warp Core.", Language.English);
            nlp.ProcessSingle(doc);
            var captured = Captured(doc, "Linked");
            Assert.False(captured.ContainsKey("Warp Coil"));
            Assert.Equal(node, captured["Warp Core"]);
        }

        [Fact]
        public async Task Spotter_RemovesAnEntryWithoutRebuilding()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("Curiosity");
            spotter.AddEntry("New York");
            spotter.Flush();

            spotter.RemoveEntry("Curiosity");
            spotter.Flush();

            var doc = new Document("Curiosity is in New York.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var values = doc.SelectMany(s => s.GetEntities()).Select(e => e.Value).ToArray();
            Assert.Equal(new[] { "New York" }, values);
            Assert.Equal(new[] { "New York" }, spotter.GetEntries().ToArray());
        }
    }
}
