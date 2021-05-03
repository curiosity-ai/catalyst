
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static sealed class Map
        {
            private static Lazy<Lookups> _lookup_us2uk = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_us2uk.bin")).WaitResult());
            private static Lazy<Lookups> _lookup_uk2us = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_uk2us.bin")).WaitResult());

            public static string ToAmericanEnglish(IToken token) => new string(ToAmericanEnglishAsSpan(token));
            public static string ToBritishEnglish(IToken token) => new string(ToBritishEnglishAsSpan(token));
            public static ReadOnlySpan<char> ToAmericanEnglishAsSpan(IToken token) => _lookup_uk2us.Value.Get(token);
            public static ReadOnlySpan<char> ToBritishEnglishAsSpan(IToken token) => _lookup_us2uk.Value.Get(token);
        }
    }
}
