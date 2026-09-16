
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Serbian
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Serbian;

            private static Lazy<Lookups> _lookup = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "sr_lemma_lookup.bin")).WaitResult(), isThreadSafe:true);

            public string GetLemma(ReadOnlySpan<char> value) => new string(GetLemmaAsSpan(value));

            public ReadOnlySpan<char> GetLemmaAsSpan(ReadOnlySpan<char> value) => _lookup.Value.Get(value);


            public bool IsBaseForm(ReadOnlySpan<char> value)
            {
                return false;
            }
        }
    }
}
