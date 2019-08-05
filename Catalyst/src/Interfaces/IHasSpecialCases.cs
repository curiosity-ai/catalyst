using System.Collections.Generic;

namespace Catalyst.Models
{
    internal interface IHasSpecialCases
    {
        IEnumerable<KeyValuePair<int, TokenizationException>> GetSpecialCases();
    }
}