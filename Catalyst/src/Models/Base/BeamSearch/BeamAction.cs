// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst.Models
{
    public class BeamAction
    {
        public float DeltaScore;
        public int Index;

        public BeamAction(int index, float delta)
        {
            Index = index; DeltaScore = delta;
        }
    }
}