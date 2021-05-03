
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
            private static Lazy<Lookups> _lookup_us2uk = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_us2uk.bin")).WaitResult());
            private static Lazy<Lookups> _lookup_uk2us = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_uk2us.bin")).WaitResult());

            public static string ToAmerican(IToken token) => new string(ToAmericanAsSpan(token));
            public static string ToBritish(IToken token) => new string(ToBritishAsSpan(token));
            public static ReadOnlySpan<char> ToAmericanAsSpan(IToken token) => _lookup_uk2us.Value.Get(token);
            public static ReadOnlySpan<char> ToBritishAsSpan(IToken token) => _lookup_us2uk.Value.Get(token);

            public static string ToAmerican(string token) => new string(ToAmericanAsSpan(token));
            public static string ToBritish(string token) => new string(ToBritishAsSpan(token));
            public static ReadOnlySpan<char> ToAmericanAsSpan(string token) => _lookup_uk2us.Value.Get(new SingleToken(token, Language.English));
            public static ReadOnlySpan<char> ToBritishAsSpan(string token) => _lookup_us2uk.Value.Get(new SingleToken(token, Language.English));
        }
    }
}
