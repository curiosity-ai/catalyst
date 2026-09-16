using System;
using Mosaik.Core;
using System.Collections.Generic;

namespace Catalyst.DateTimeRecognition
{
    /// <summary>Small helper so a language's vocabulary reads as a list of words rather than as dictionary plumbing.</summary>
    public sealed class LexiconBuilder
    {
        private readonly List<KeyValuePair<string, TermInfo>> _words   = new List<KeyValuePair<string, TermInfo>>();
        private readonly List<KeyValuePair<string, TermInfo>> _phrases = new List<KeyValuePair<string, TermInfo>>();

        public void Add(TermKind kind, int value, params string[] words)
        {
            Add(new TermInfo(kind, value), words);
        }

        public void Add(TermKind kind, params string[] words)
        {
            Add(new TermInfo(kind, 0), words);
        }

        public void Add(TermInfo info, params string[] words)
        {
            foreach (var w in words)
            {
                // The lexer never produces a word containing a hyphen, so "sexta-feira" is a phrase the
                // phrase folder matches across the dash - which also makes the spaced spelling work.
                var normalized = w.IndexOf('-') >= 0 ? w.Replace('-', ' ') : w;

                if (normalized.IndexOf(' ') >= 0) { _phrases.Add(new KeyValuePair<string, TermInfo>(normalized, info)); }
                else                              { _words.Add(new KeyValuePair<string, TermInfo>(normalized, info)); }
            }
        }

        public Lexicon Build(Language language, bool dayMonthOrder, bool decimalComma = false, bool articleInDateSpan = true, bool articleInPeriodSpan = false, bool relativeAfterUnit = false, bool pluralEndsInS = true, bool partNamedWithOf = false, bool minutesFollowHour = false, bool splitsCompounds = false, bool halfIsBeforeTheHour = false, bool ordinalEndsInDot = false, bool movableHolidayNamesItsDay = true) => new Lexicon(language, dayMonthOrder, _words, _phrases, decimalComma, articleInDateSpan, articleInPeriodSpan, relativeAfterUnit, pluralEndsInS, partNamedWithOf, minutesFollowHour, splitsCompounds, halfIsBeforeTheHour, ordinalEndsInDot, movableHolidayNamesItsDay);
    }
}
