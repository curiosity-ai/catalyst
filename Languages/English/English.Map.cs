
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public sealed class Map
        {
            public Language Language => Language.English;
            
            private static Lazy<Lookups> _lookup_us2uk = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_us2uk.bin")).WaitResult());
            private static Lazy<Lookups> _lookup_uk2us = new Lazy<Lookups>(() => Lookups.FromStream(ResourceLoader.OpenResource(typeof(Lemmatizer).Assembly, "en_uk2us.bin")).WaitResult());

            public string GetLemma(IToken token) => new string(ToAmericanEnglish(token));
            
            public ReadOnlySpan<char> ToAmericanEnglish(IToken token) => _lookup_uk2us.Value.Get(token);
            public ReadOnlySpan<char> ToBritishEnglish(IToken token) => _lookup_us2uk.Value.Get(token);
        }
    }
}
