
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Tamil
    {
        internal sealed class TokenizerExceptions 
        {
            internal static Dictionary<int, TokenizationException> Get()
            {
                var exceptions = Catalyst.TokenizerExceptions.CreateBaseExceptions();
                return exceptions;
            }
        }
    }
}
