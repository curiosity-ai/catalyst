using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        private static object _lockSpanishExceptions = new object();
        private static Dictionary<int, TokenizationException> SpanishExceptions;

        private static Dictionary<int, TokenizationException> GetSpanishExceptions()
        {
            if (SpanishExceptions is null)
            {
                lock (_lockSpanishExceptions)
                {
                    if (SpanishExceptions is null)
                    {
                        SpanishExceptions = CreateBaseExceptions();

                        Create(SpanishExceptions, "", "pal|pala", "para el|para la");
                        Create(SpanishExceptions, "", "aprox.|dna.|esq.|pág.|p.ej.|Ud.|Vd.|Uds.|Vds.", "aproximadamente|docena|esquina|página|por ejemplo|usted|usted|ustedes|ustedes");
                        Create(SpanishExceptions, "", "12m.", "12 p.m.");
                        Create(SpanishExceptions, "", "a.C.|a.J.C.|apdo.|Av.|Avda.|Cía.|etc.|Gob.|Gral.|Ing.|J.C.|Lic.|m.n.|no.|núm.|P.D.|Prof.|Profa.|q.e.p.d.|S.A.|S.L.|s.s.s.|Sr.|Sra.|Srta.", "a.C.|a.J.C.|apdo.|Av.|Avda.|Cía.|etc.|Gob.|Gral.|Ing.|J.C.|Lic.|m.n.|no.|núm.|P.D.|Prof.|Profa.|q.e.p.d.|S.A.|S.L.|s.s.s.|Sr.|Sra.|Srta.");
                    }
                }
            }
            return SpanishExceptions;
        }
    }
}
