
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Italian
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Italian;

            private static Lazy<Lookups> _lookup = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "it_lemma_lookup.bin")).WaitResult(), isThreadSafe:true);

            public string GetLemma(IToken token) => new string(GetLemmaAsSpan(token));

            public ReadOnlySpan<char> GetLemmaAsSpan(IToken token) => _lookup.Value.Get(token);


            public bool IsBaseForm(IToken token)
            {
                return false;
            }
        }
    }
}
