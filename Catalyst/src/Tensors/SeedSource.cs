using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors
{
    public class SeedSource
    {
        private Random rng;

        public SeedSource()
        {
            rng = new Random();
        }

        public SeedSource(int seed)
        {
            rng = new Random();
        }

        public void SetSeed(int seed)
        {
            rng = new Random(seed);
        }

        public int NextSeed()
        {
            return rng.Next();
        }
    }
}
