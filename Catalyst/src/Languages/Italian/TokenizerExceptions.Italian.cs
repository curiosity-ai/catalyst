using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        private static object _lockItalianExceptions = new object();
        private static Dictionary<int, TokenizationException> ItalianExceptions;

        private static Dictionary<int, TokenizationException> GetItalianExceptions()
        {
            if (ItalianExceptions is null)
            {
                lock (_lockItalianExceptions)
                {
                    if (ItalianExceptions is null)
                    {
                        ItalianExceptions = CreateBaseExceptions();

                        Create(ItalianExceptions, "", "po'", "poco");
                    }
                }
            }
            return ItalianExceptions;
        }
    }
}
