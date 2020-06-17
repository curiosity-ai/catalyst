using Mosaik.Core;
using System;
using System.Text;

namespace Catalyst
{
    internal static class Pools
    {
        internal static ObjectPool<StringBuilder> StringBuilder = new ObjectPool<StringBuilder>(() => new StringBuilder(), Environment.ProcessorCount, sb => sb.Clear());
    }
}