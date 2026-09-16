
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Chinese
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Chinese;

            public string GetLemma(ReadOnlySpan<char> value)
            {
                return new string(GetLemmaAsSpan(value));
            }

            public ReadOnlySpan<char> GetLemmaAsSpan(ReadOnlySpan<char> value)
            {
                return value;
            }

            public bool IsBaseForm(ReadOnlySpan<char> value)
            {
                return false;
            }
        }
    }
}
