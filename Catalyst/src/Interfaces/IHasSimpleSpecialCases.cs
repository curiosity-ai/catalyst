using System.Collections.Generic;

namespace Catalyst.Models
{
    internal interface IHasSimpleSpecialCases
    {
        IEnumerable<int> GetSimpleSpecialCases();
    }
}