using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Catalyst.Models;
using MessagePack;
using Mosaik.Core;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    // The stored form of a spotter changed: entries are kept as a prefix-compressed dictionary of their
    // surface forms rather than as hashes. These tests cover the round trip, and that a model written in the
    // old hash format still loads and matches exactly as it did - hashes cannot be turned back into strings,
    // so such a model keeps using the old tables for its whole life.
    public class SpotterModelFormatTests
    {
        private static readonly MessagePackSerializerOptions Lz4Standard = MessagePackSerializerOptions.Standard.WithCompression(MessagePackCompression.Lz4Block);

        private static string[] EntityValues(IDocument doc) =>
            doc.SelectMany(span => span.GetEntities()).Select(e => e.Value).OrderBy(v => v).ToArray();

        private static async Task<Pipeline> PipelineWithAsync(IProcess process)
        {
            English.Register();
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false, sentenceDetector: false);
            nlp.Add(process);
            return nlp;
        }

        private const string TEXT = "Fit NAS1291-C3M next to the San Francisco Bay office in New York.";

        [Fact]
        public async Task Spotter_SurvivesAStoreAndReload()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("NAS1291-C3M");
            spotter.AddEntry("New York");
            spotter.AddEntry("San Francisco Bay");
            spotter.AddEntry("AT&T");

            var before = new Document(TEXT, Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(before);

            using var stream = new MemoryStream();
            await spotter.StoreAsync(stream);
            stream.Seek(0, SeekOrigin.Begin);

            var reloaded = new Spotter(Language.English, 0, "", "Entity");
            await reloaded.LoadAsync(stream);
            reloaded.TrimExcess();

            Assert.False(reloaded.IsLegacyModel);
            Assert.Equal(new[] { "AT&T", "NAS1291-C3M", "New York", "San Francisco Bay" }, reloaded.GetEntries().OrderBy(e => e).ToArray());

            var after = new Document(TEXT, Language.English);
            (await PipelineWithAsync(reloaded)).ProcessSingle(after);

            Assert.Equal(EntityValues(before), EntityValues(after));
            Assert.Contains("San Francisco Bay", EntityValues(after));
            Assert.True(spotter.IsEquivalentTo(reloaded));
        }

        [Fact]
        public async Task LinkedSpotter_SurvivesAStoreAndReloadWithItsValues()
        {
            var bay      = UID128.New();
            var newYork  = UID128.New();
            var part     = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("San Francisco Bay", bay);
            spotter.AddEntry("New York", newYork);
            spotter.AddEntry("NAS1291-C3M", part);

            using var stream = new MemoryStream();
            await spotter.StoreAsync(stream);
            stream.Seek(0, SeekOrigin.Begin);

            var reloaded = new LinkedSpotter(Language.English, 0, "", "Linked");
            await reloaded.LoadAsync(stream);
            reloaded.TrimExcess();

            Assert.False(reloaded.IsLegacyModel);

            var doc = new Document(TEXT, Language.English);
            (await PipelineWithAsync(reloaded)).ProcessSingle(doc);

            var captured = doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == "Linked").ToDictionary(e => e.Value, e => e.EntityType.TargetUID);
            Assert.Equal(bay,     captured["San Francisco Bay"]);
            Assert.Equal(newYork, captured["New York"]);
            Assert.Equal(part,    captured["NAS1291-C3M"]);
        }

        [Fact]
        public async Task Spotter_StoredInTheOldHashFormatStillLoadsAndMatches()
        {
            // Hand-built in the pre-dictionary shape: whole-entry hashes plus one set per word position.
            var model = new SpotterModel { CaptureTag = "Entity", MinTokenLength = 3, MaxTokenLength = 11 };

            model.Hashes.Add(Spotter.Hash64("Curiosity".AsSpan()));

            var first  = Spotter.Hash64("New".AsSpan());
            var second = Spotter.Hash64("York".AsSpan());
            model.MultiGramHashes.Add(new HashSet<ulong> { first });
            model.MultiGramHashes.Add(new HashSet<ulong> { second });
            model.Hashes.Add(Spotter.HashCombine64(first, second));

            using var stream = new MemoryStream(MessagePackSerializer.Serialize(model, Lz4Standard));

            var spotter = new Spotter(Language.English, 0, "", "Entity");
            await spotter.LoadAsync(stream);
            spotter.TrimExcess();

            Assert.True(spotter.IsLegacyModel);
            Assert.True(spotter.IsMemoryOptimized);

            var doc = new Document("Curiosity works in New York.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var values = EntityValues(doc);
            Assert.Contains("Curiosity", values);
            Assert.Contains("New York", values);
        }

        [Fact]
        public async Task LinkedSpotter_StoredInTheOldHashFormatStillLoadsAndMatches()
        {
            var curiosity = UID128.New();
            var newYork   = UID128.New();

            var model = new LinkedSpotterModel { CaptureTag = "Linked", MinTokenLength = 3, MaxTokenLength = 11 };
            model.Hashes[Spotter.Hash64("Curiosity".AsSpan())] = curiosity;

            var first  = Spotter.Hash64("New".AsSpan());
            var second = Spotter.Hash64("York".AsSpan());
            model.MultiGramHashes.Add(new HashSet<ulong> { first });
            model.MultiGramHashes.Add(new HashSet<ulong> { second });
            model.Hashes[Spotter.HashCombine64(first, second)] = newYork;

            using var stream = new MemoryStream(MessagePackSerializer.Serialize(model, Lz4Standard));

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            await spotter.LoadAsync(stream);
            spotter.TrimExcess();

            Assert.True(spotter.IsLegacyModel);

            var doc = new Document("Curiosity works in New York.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var captured = doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == "Linked").ToDictionary(e => e.Value, e => e.EntityType.TargetUID);
            Assert.Equal(curiosity, captured["Curiosity"]);
            Assert.Equal(newYork,   captured["New York"]);
        }

        [Fact]
        public async Task Spotter_ReOpensForEditingAndKeepsWhatItHad()
        {
            var spotter = new Spotter(Language.English, 0, "", "Entity");
            spotter.AddEntry("San Francisco Bay");
            spotter.TrimExcess();
            Assert.True(spotter.IsMemoryOptimized);

            spotter.AddEntry("New York");
            spotter.AddEntry("NAS1291-C3M");

            var doc = new Document(TEXT, Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var values = EntityValues(doc);
            Assert.Contains("San Francisco Bay", values);
            Assert.Contains("New York", values);
            Assert.Contains("NAS1291-C3M", values);
        }

        [Fact]
        public async Task LinkedSpotter_ReOpensForEditingAndKeepsEveryValueAlignedWithItsEntry()
        {
            var bay     = UID128.New();
            var newYork = UID128.New();
            var part    = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("San Francisco Bay", bay);
            spotter.TrimExcess();

            // Re-opening has to keep every value with the entry it was added for, even though the entries are
            // re-sorted around the new ones.
            spotter.AddEntry("New York", newYork);
            spotter.AddEntry("NAS1291-C3M", part);

            var doc = new Document(TEXT, Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var captured = doc.SelectMany(s => s.GetEntities()).Where(e => e.EntityType.Type == "Linked").ToDictionary(e => e.Value, e => e.EntityType.TargetUID);
            Assert.Equal(bay,     captured["San Francisco Bay"]);
            Assert.Equal(newYork, captured["New York"]);
            Assert.Equal(part,    captured["NAS1291-C3M"]);
        }

        [Fact]
        public async Task LinkedSpotter_LastValueWinsWhenTheSameEntryIsAddedTwice()
        {
            var first  = UID128.New();
            var second = UID128.New();

            var spotter = new LinkedSpotter(Language.English, 0, "", "Linked");
            spotter.AddEntry("New York", first);
            spotter.AddEntry("New York", second);

            var doc = new Document("Visiting New York soon.", Language.English);
            (await PipelineWithAsync(spotter)).ProcessSingle(doc);

            var entity = doc.SelectMany(s => s.GetEntities()).Single(e => e.EntityType.Type == "Linked");
            Assert.Equal(second, entity.EntityType.TargetUID);
        }
    }
}
