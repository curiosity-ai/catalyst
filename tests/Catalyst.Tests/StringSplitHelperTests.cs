using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace Catalyst.Tests
{
    public class StringSplitHelperTests
    {
        [Fact]
        public void PerfectMatchSplitsCorrectly()
        {
            var original = "Hello friend! This is a test email.\n\nBest regards,\nJohn Doe";
            var parts = new List<string> {
                "Hello friend! ",
                "This is a test email.\n\n",
                "Best regards,\nJohn Doe"
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(3, splits.Count);
            Assert.Equal("Hello friend! ", splits[0]);
            Assert.Equal("This is a test email.\n\n", splits[1]);
            Assert.Equal("Best regards,\nJohn Doe", splits[2]);
            Assert.Equal(original, string.Join("", splits));
        }

        [Fact]
        public void SplitsWithTyposAndWhitespaceChanges()
        {
            var original = "1) Hello friend! This is a test email.\n\nBest regards,\nJohn Doe";
            var parts = new List<string> {
                "Hllo my firend",
                "This is test email.",
                "Best rgrds, John Doe"
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(3, splits.Count);
            Assert.Equal("1) Hello friend! ", splits[0]);
            Assert.Equal("This is a test email.\n\n", splits[1]);
            Assert.Equal("Best regards,\nJohn Doe", splits[2]);
            Assert.Equal(original, string.Join("", splits));
        }

        [Fact]
        public void ComplexEmailWithInlineReplies()
        {
            var original = @"Hi Sarah,

> Can we meet tomorrow at 10 AM?
Yes, that works for me.

> Also, please bring the quarterly report.
I have it ready and will bring a printed copy.

Thanks,
Mike";

            var parts = new List<string>
            {
                "Hi Sarah,",
                "Can we meet tomorrow at 10 AM? Yes, that works for me.",
                "Also, please bring the quarterly report. I have it ready and will bring a printed copy.",
                "Thanks, Mike"
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(4, splits.Count);
            // Splitting should correctly segment the whole string
            Assert.Equal("Hi Sarah,\n\n> ".ReplaceLineEndings(), splits[0].ReplaceLineEndings());
            Assert.Equal("Can we meet tomorrow at 10 AM?\nYes, that works for me.\n\n> ".ReplaceLineEndings(), splits[1].ReplaceLineEndings());
            Assert.Equal("Also, please bring the quarterly report.\nI have it ready and will bring a printed copy.\n\n".ReplaceLineEndings(), splits[2].ReplaceLineEndings());
            Assert.Equal("Thanks,\nMike".ReplaceLineEndings(), splits[3].ReplaceLineEndings());
            Assert.Equal(original, string.Join("", splits));
        }

        [Fact]
        public void RepetitionOfContent()
        {
            var original = "Testing test test test. And again test test test. Finally test test.";
            var parts = new List<string> {
                "Testing test test tst.",
                "And again test test tset.",
                "Finaly test test."
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(3, splits.Count);
            Assert.Equal("Testing test test test. ", splits[0]);
            Assert.Equal("And again test test test. ", splits[1]);
            Assert.Equal("Finally test test.", splits[2]);
            Assert.Equal(original, string.Join("", splits));
        }

        [Fact]
        public void MissingWordsInParts()
        {
            var original = "The quick brown fox jumps over the lazy dog.";
            var parts = new List<string> {
                "The brown fox",
                "jumps over lazy dog."
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(2, splits.Count);
            Assert.Equal("The quick brown fox ", splits[0]);
            Assert.Equal("jumps over the lazy dog.", splits[1]);
            Assert.Equal(original, string.Join("", splits));
        }

        [Fact]
        public void InvalidInput()
        {
            var original = "The quick brown fox jumps over the lazy dog.";
            var parts = new List<string> {
                "The tomato",
                "is a fruit."
            };

            var splits = StringSplitHelper.SplitPerfectly(original, parts).ToList();

            Assert.Equal(0, splits.Count);
        }
    }
}
