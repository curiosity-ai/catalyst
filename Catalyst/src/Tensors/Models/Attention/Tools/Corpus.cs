using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public class TokenPairs
    {
        public string[] Source;
        public string[] Target;
    }

    public class Corpus : IEnumerable<TokenPairs>
    {
        private static ILogger Logger = ApplicationLogging.CreateLogger<Corpus>();

        public IEnumerable<ISpan[]> SentencePairs { get; }
        public Func<IToken, string> SourceSelector { get; }
        public Func<IToken, string> TargetSelector { get; }

        public IEnumerator<TokenPairs> GetEnumerator()
        {
            var list = new List<TokenPairs>();

            foreach(var pair in SentencePairs)
            {
                if(pair.Length != 2)
                {
                    throw new Exception($"Invalid document pair found. Expected Length = 2, found {pair.Length} instead");
                }

                list.Add(new TokenPairs()
                {
                    Source = pair[0].Select(t => SourceSelector(t)).ToArray(),
                    Target = pair[1].Select(t => SourceSelector(t)).ToArray()
                });


                if(list.Count > 1_000)
                {
                    ShuffleByLength(list);
                    foreach(var tp in list) { yield return tp; }
                    list.Clear();
                }
            }

            if (list.Count > 0)
            {
                ShuffleByLength(list);
                foreach (var tp in list) { yield return tp; }
            }
        }

        private void ShuffleByLength(List<TokenPairs> list)
        {
            list.Shuffle(); //Shuffle first, so the sort order bellow is random
            list.Sort(new TokenPairCompareByLength());
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public Corpus(IEnumerable<ISpan[]> sentences, Func<IToken, string> sourceSelector = null, Func<IToken, string> targetSelector = null)
        {
            SourceSelector = sourceSelector ?? ((t) => t.Value);
            TargetSelector = targetSelector ?? ((t) => t.Value);

            SentencePairs = sentences;
        }

        private class TokenPairCompareByLength : IComparer<TokenPairs>
        {
            public int Compare(TokenPairs x, TokenPairs y)
            {
                return x.Source.Length.CompareTo(y.Source.Length);
            }
        }
    }
}
