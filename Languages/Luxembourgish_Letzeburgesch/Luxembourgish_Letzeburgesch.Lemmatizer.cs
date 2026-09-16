
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Luxembourgish_Letzeburgesch
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Luxembourgish_Letzeburgesch;

            private static Lazy<Lookups> _lookup = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "lb_lemma_lookup.bin")).WaitResult(), isThreadSafe:true);

            public string GetLemma(ReadOnlySpan<char> value) => new string(GetLemmaAsSpan(value));

            public ReadOnlySpan<char> GetLemmaAsSpan(ReadOnlySpan<char> value) => _lookup.Value.Get(value);


            public bool IsBaseForm(ReadOnlySpan<char> value)
            {
                return false;
            }
        }
    }
}
