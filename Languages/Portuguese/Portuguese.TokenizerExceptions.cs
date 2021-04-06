
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class Portuguese
    {
        internal sealed class TokenizerExceptions 
        {
            internal static Dictionary<int, TokenizationException> Get()
            {
                var exceptions = Catalyst.TokenizerExceptions.CreateBaseExceptions();
                
                Catalyst.TokenizerExceptions.Create(exceptions, "", "Adm.|Dr.|e.g.|E.g.|E.G.|Gen.|Gov.|i.e.|I.e.|I.E.|Jr.|Ltd.|p.m.|Ph.D.|Rep.|Rev.|Sen.|Sr.|Sra.|vs.|tel.|pág.|pag.", "Adm.|Dr.|e.g.|E.g.|E.G.|Gen.|Gov.|i.e.|I.e.|I.E.|Jr.|Ltd.|p.m.|Ph.D.|Rep.|Rev.|Sen.|Sr.|Sra.|vs.|tel.|pág.|pag.");

                return exceptions;
            }
        }
    }
}
