
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Czech
    {
        public static class StopWords
        {
            public static ReadOnlyHashSet<string> Snowball = new ReadOnlyHashSet<string>(new HashSet<string>(new string[] { }));
            public static ReadOnlyHashSet<string> Spacy    = new ReadOnlyHashSet<string>(new HashSet<string>(new string[] { }));
        }
    }
}
