using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        private static object _lockPortugueseExceptions = new object();
        private static Dictionary<int, TokenizationException> PortugueseExceptions;

        private static Dictionary<int, TokenizationException> GetPortugueseExceptions()
        {
            if (PortugueseExceptions is null)
            {
                lock (_lockPortugueseExceptions)
                {
                    if (PortugueseExceptions is null)
                    {
                        PortugueseExceptions = BaseExceptions();

                        Create(PortugueseExceptions, "", "Adm.|Dr.|e.g.|E.g.|E.G.|Gen.|Gov.|i.e.|I.e.|I.E.|Jr.|Ltd.|p.m.|Ph.D.|Rep.|Rev.|Sen.|Sr.|Sra.|vs.|tel.|pág.|pag.", "Adm.|Dr.|e.g.|E.g.|E.G.|Gen.|Gov.|i.e.|I.e.|I.E.|Jr.|Ltd.|p.m.|Ph.D.|Rep.|Rev.|Sen.|Sr.|Sra.|vs.|tel.|pág.|pag.");
                    }
                }
            }
            return PortugueseExceptions;
        }
    }
}
