// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System.Collections.Generic;

namespace Catalyst.Models
{
    internal interface IHasSpecialCases
    {
        IEnumerable<KeyValuePair<int, TokenizationException>> GetSpecialCases();
    }
}