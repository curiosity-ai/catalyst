// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst.Models
{

    internal struct SplitPoint
    {
        internal int Begin;
        internal int End;
        internal SplitPointReason Reason;

        internal SplitPoint(int b, int e, SplitPointReason reason)
        {
            Begin = b; End = e; Reason = reason;
        }
    }
}