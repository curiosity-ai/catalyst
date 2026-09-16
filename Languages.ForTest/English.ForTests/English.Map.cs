
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static class Map
        {
            private static Lazy<Lookups> _lookup_us2uk = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_us2uk.bin")).WaitResult(), isThreadSafe:true);
            private static Lazy<Lookups> _lookup_uk2us = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_uk2us.bin")).WaitResult(), System.Threading.LazyThreadSafetyMode.ExecutionAndPublication);

            public static string ToAmerican(ReadOnlySpan<char> value) => new string(ToAmericanAsSpan(value));
            public static string ToBritish(ReadOnlySpan<char> value) => new string(ToBritishAsSpan(value));
            public static ReadOnlySpan<char> ToAmericanAsSpan(ReadOnlySpan<char> value) => _lookup_uk2us.Value.Get(value);
            public static ReadOnlySpan<char> ToBritishAsSpan(ReadOnlySpan<char> value) => _lookup_us2uk.Value.Get(value);
        }
    }
}
