// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using Mosaik.Core;

namespace Catalyst
{
    public interface IProcess : IModel
    {
        void Process(IDocument document);
    }
}