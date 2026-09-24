using System.Linq;
using Catalyst.Models;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests
{
    /// <summary>
    /// An emoji glued to a word is its own token, and the word keeps every letter. The split point that
    /// separates them is not a character (unlike whitespace), so the tokenizer must not skip one when it
    /// moves on to the emoji, and must not shorten the word to compensate.
    /// </summary>
    public class TokenizerEmojiTests
    {
        private static string[] Tokens(string text)
        {
            var document = new Document(text, Language.English);
            new FastTokenizer(Language.English).Parse(document);
            return document.Spans.SelectMany(s => s.Tokens).Select(t => t.Value).ToArray();
        }

        [Theory]
        [InlineData("pn5364door😀",     new[] { "pn5364door", "😀" })]
        [InlineData("😀pn5364door",     new[] { "😀", "pn5364door" })]
        [InlineData("door😀door",       new[] { "door", "😀", "door" })]
        [InlineData("door😀😀",         new[] { "door", "😀", "😀" })]
        [InlineData("door!😀",          new[] { "door", "!", "😀" })]
        [InlineData("one😀 two😀three", new[] { "one", "😀", "two", "😀", "three" })]
        [InlineData("a😀",              new[] { "a", "😀" })]
        public void AnEmojiGluedToAWordLeavesTheWordWhole(string text, string[] expected)
        {
            Assert.Equal(expected, Tokens(text));
        }
    }
}
