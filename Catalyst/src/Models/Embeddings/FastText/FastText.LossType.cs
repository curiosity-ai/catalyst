// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst.Models
{
    public partial class FastText
    {
        public enum LossType
        {
            HierarchicalSoftMax,
            SoftMax,
            NegativeSampling
        }
    }
}