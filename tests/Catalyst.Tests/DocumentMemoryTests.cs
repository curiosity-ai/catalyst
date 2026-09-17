using MessagePack;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Runtime.InteropServices;
using Xunit;

namespace Catalyst.Tests
{
    /// <summary>
    /// A document stores its text as <see cref="ReadOnlyMemory{T}"/>, so it can be a view over a buffer the
    /// caller already holds rather than a copy of it. These tests measure that the view really is a view -
    /// that nothing along the read path quietly copies the text back into a string.
    /// </summary>
    public class DocumentMemoryTests
    {
        private const string Text   = "Curiosity GmbH is based in Berlin .";
        private const int    Offset = 10;

        //Text sitting inside a larger buffer, the way a document carved out of a bigger file would be
        private static readonly string Buffer = new string('.', Offset) + Text + new string('!', 17);

        private static long Measure(Action action, int iterations)
        {
            for (int i = 0; i < 50; i++) action(); // Warm up, so the JIT is not what gets measured

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long before = GC.GetAllocatedBytesForCurrentThread();

            for (int i = 0; i < iterations; i++) action();

            return (GC.GetAllocatedBytesForCurrentThread() - before) / iterations;
        }

        private static void Tokenize(IDocument document)
        {
            var span = document.AddSpan(0, Text.Length - 1);

            int at = 0;
            foreach (var word in Text.Split(' '))
            {
                span.AddTokenAsStruct(at, at + word.Length - 1);
                at += word.Length + 1;
            }
        }

        private static Document NewSliceBacked()
        {
            var document = new Document(Buffer.AsMemory(Offset, Text.Length), Language.English);
            Tokenize(document);
            return document;
        }

        [Fact]
        public void ADocumentOverASliceIsAViewOverTheOriginalBuffer()
        {
            var document = NewSliceBacked();

            Assert.True(MemoryMarshal.TryGetString(document.ValueMemory, out var backing, out int start, out int length));

            Assert.Same(Buffer, backing);
            Assert.Equal(Offset, start);
            Assert.Equal(Text.Length, length);
        }

        [Fact]
        public void ADocumentOverAStringHandsThatSameStringBack()
        {
            var text     = string.Concat(Text, ""); //A fresh instance, so Assert.Same means identity and not interning
            var document = new Document(text, Language.English);

            Assert.Same(text, document.Value);
        }

        [Fact]
        public void ASliceBackedDocumentReadsItsTextAndTokens()
        {
            var document = NewSliceBacked();

            Assert.Equal(Text,        document.Value);
            Assert.Equal(Text.Length, document.Length);
            Assert.Equal(Text,        document.ValueAsSpan.ToString());

            var words  = Text.Split(' ');
            var tokens = document.ToTokenList();

            Assert.Equal(words.Length, tokens.Count);

            for (int i = 0; i < words.Length; i++)
            {
                Assert.Equal(words[i], tokens[i].Value);
                Assert.Equal(words[i], tokens[i].ValueAsSpan.ToString());
            }
        }

        [Fact]
        public void MaterializingASliceBackedValueHappensOnceAndIsCached()
        {
            var document = NewSliceBacked();

            var first  = document.Value;
            var second = document.Value;

            Assert.Same(first, second);
        }

        [Fact]
        public void ReadingASliceBackedDocumentThroughSpansAllocatesNothing()
        {
            var document = NewSliceBacked();
            int total    = 0;

            long perCall = Measure(() =>
            {
                total += document.ValueAsSpan.Length;
                total += document.ValueMemory.Length;
                total += document.GetTokenValueAsSpan(0, 0).Length;
                total += document.GetSpanValue2(0).Length;
            }, iterations: 5000);

            Assert.True(total > 0);
            Assert.True(perCall == 0, $"expected an allocation-free read, measured {perCall} bytes per call");
        }

        [Fact]
        public void ASliceBackedDocumentSerializesToTheSameBytesAsAStringBackedOne()
        {
            var fromString = new Document(Text, Language.English);
            Tokenize(fromString);

            var pool   = new DocumentPool();
            var pooled = pool.Rent(Buffer.AsMemory(Offset, Text.Length), Language.English);
            Tokenize(pooled);

            var buffer = new ArrayBufferWriter<byte>();
            pooled.SerializeAsMessagePack(buffer);

            Assert.Equal(MessagePackSerializer.Serialize(fromString), buffer.WrittenSpan.ToArray());

            pool.Return(pooled);
        }

        [Fact]
        public void ASliceBackedDocumentStaysAViewAcrossTheImmutableRoundTrip()
        {
            var immutable = NewSliceBacked().ToImmutable();

            Assert.True(MemoryMarshal.TryGetString(immutable.ValueMemory, out var backing, out int start, out _));
            Assert.Same(Buffer, backing);
            Assert.Equal(Offset, start);

            var mutable = immutable.ToMutable();

            Assert.True(MemoryMarshal.TryGetString(mutable.ValueMemory, out var roundTripped, out int roundTrippedStart, out _));
            Assert.Same(Buffer, roundTripped);
            Assert.Equal(Offset, roundTrippedStart);
            Assert.Equal(Text, mutable.Value);
        }

        [Fact]
        public void APooledDocumentRentedOverMemoryIsAViewOverTheOriginalBuffer()
        {
            var pool     = new DocumentPool();
            var document = pool.Rent(Buffer.AsMemory(Offset, Text.Length), Language.English);

            Assert.True(MemoryMarshal.TryGetString(document.ValueMemory, out var backing, out int start, out _));
            Assert.Same(Buffer, backing);
            Assert.Equal(Offset, start);
            Assert.Equal(Text, document.Value);

            pool.Return(document);
        }

        [Fact]
        public void TheSpanMembersDoNotBreakJsonSerialization()
        {
            //Document is a [JsonObject], so a public ReadOnlySpan member would be reflected over unless ignored
            var json = Newtonsoft.Json.JsonConvert.SerializeObject(NewSliceBacked());

            Assert.Contains(Text, json);
            Assert.DoesNotContain(nameof(Document.ValueAsSpan), json);
            Assert.DoesNotContain(nameof(Document.ValueMemory), json);
        }

        [Fact]
        public void ANullValueStaysNull()
        {
            var document = new Document(Text, Language.English) { Value = null };

            Assert.Null(document.Value);
            Assert.True(document.IsValueNull);

            document.Value = Text;

            Assert.False(document.IsValueNull);
            Assert.Equal(Text, document.Value);
        }

        [Fact]
        public void ControlCharactersAreRemovedFromMemoryTheSameWayAsFromAString()
        {
            var dirty = "Curiosity" + (char)1 + " GmbH";

            Assert.Equal(new Document(dirty).Value, new Document(dirty.AsMemory()).Value);
        }

        [Fact]
        public void CleanMemoryIsNotCopiedWhileRemovingControlCharacters()
        {
            var document = new Document(Buffer.AsMemory(Offset, Text.Length));

            //Nothing to remove, so the document must still point at the caller's buffer
            Assert.True(MemoryMarshal.TryGetString(document.ValueMemory, out var backing, out _, out _));
            Assert.Same(Buffer, backing);
        }
    }
}
