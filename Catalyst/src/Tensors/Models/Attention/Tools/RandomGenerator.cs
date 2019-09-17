using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors
{
    public static class RandomGenerator
    {

        public static bool Return_V { get; set; }
        public static float V_Val { get; set; }

        public static float GaussRandom()
        {
            if (Return_V)
            {
                Return_V = false;
                return V_Val;
            }

            Span<float> floats = stackalloc float[2];

            ThreadSafeFastRandom.NextFloats(floats);

            var u = 2 * floats[0] - 1;
            var v = 2 * floats[1] - 1;
            var r = (u * u) + (v * v);

            if (r == 0 || r > 1) return GaussRandom();
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            V_Val = (float)(v * c);
            Return_V = true;
            return (float)(u * c);
        }

        public static float NormalRandom(float mu, float std)
        {
            return mu + GaussRandom() * std;
        }

    }
     
}
