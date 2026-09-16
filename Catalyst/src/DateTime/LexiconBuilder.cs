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
                if (w.IndexOf(' ') >= 0) { _phrases.Add(new KeyValuePair<string, TermInfo>(w, info)); }
                else                     { _words.Add(new KeyValuePair<string, TermInfo>(w, info)); }
            }
        }

        public Lexicon Build(Language language, bool dayMonthOrder) => new Lexicon(language, dayMonthOrder, _words, _phrases);
    }
}
