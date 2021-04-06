
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Tagalog
    {
        internal sealed class Lemmatizer : ILemmatizer
        {
            public Language Language => Language.Tagalog;

            public string GetLemma(IToken token)
            {
                return token.Value;
            }

            public ReadOnlySpan<char> GetLemmaAsSpan(IToken token)
            {
                return token.ValueAsSpan;
            }

            public bool IsBaseForm(IToken token)
            {
                return false;
            }
        }
    }
}
