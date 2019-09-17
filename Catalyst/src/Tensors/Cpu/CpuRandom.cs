using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    [OpsClass]
    public class CpuRandom
    {
        private static readonly Random seedGen = new Random();


        public CpuRandom()
        {
        }


        // allArgs should start with a null placeholder for the RNG object
        private static void InvokeWithRng(int? seed, MethodInfo method, params object[] allArgs)
        {
            if (!seed.HasValue)
                seed = seedGen.Next();

            IntPtr rng;
            NativeWrapper.CheckResult(CpuOpsNative.TS_NewRNG(out rng));
            NativeWrapper.CheckResult(CpuOpsNative.TS_SetRNGSeed(rng, seed.Value));
            allArgs[0] = rng;
            NativeWrapper.InvokeTypeMatch(method, allArgs);
            NativeWrapper.CheckResult(CpuOpsNative.TS_DeleteRNG(rng));
        }

        private MethodInfo uniform_func = NativeWrapper.GetMethod("TS_RandomUniform");
        [RegisterOpStorageType("random_uniform", typeof(CpuStorage))]
        public void Uniform(Tensor result, int? seed, float min, float max) { InvokeWithRng(seed, uniform_func, null, result, min, max); }

        private MethodInfo normal_func = NativeWrapper.GetMethod("TS_RandomNormal");
        [RegisterOpStorageType("random_normal", typeof(CpuStorage))]
        public void Normal(Tensor result, int? seed, float mean, float stdv) { InvokeWithRng(seed, normal_func, null, result, mean, stdv); }

        private MethodInfo exponential_func = NativeWrapper.GetMethod("TS_RandomExponential");
        [RegisterOpStorageType("random_exponential", typeof(CpuStorage))]
        public void Exponential(Tensor result, int? seed, float lambda) { InvokeWithRng(seed, exponential_func, null, result, lambda); }

        private MethodInfo cauchy_func = NativeWrapper.GetMethod("TS_RandomCauchy");
        [RegisterOpStorageType("random_cauchy", typeof(CpuStorage))]
        public void Cauchy(Tensor result, int? seed, float median, float sigma) { InvokeWithRng(seed, cauchy_func, null, result, median, sigma); }

        private MethodInfo log_normal_func = NativeWrapper.GetMethod("TS_RandomLogNormal");
        [RegisterOpStorageType("random_lognormal", typeof(CpuStorage))]
        public void LogNormal(Tensor result, int? seed, float mean, float stdv) { InvokeWithRng(seed, log_normal_func, null, result, mean, stdv); }

        private MethodInfo geometric_func = NativeWrapper.GetMethod("TS_RandomGeometric");
        [RegisterOpStorageType("random_geometric", typeof(CpuStorage))]
        public void Geometric(Tensor result, int? seed, float p) { InvokeWithRng(seed, geometric_func, null, result, p); }

        private MethodInfo bernoulli_func = NativeWrapper.GetMethod("TS_RandomBernoulli");
        [RegisterOpStorageType("random_bernoulli", typeof(CpuStorage))]
        public void Bernoulli(Tensor result, int? seed, float p) { InvokeWithRng(seed, bernoulli_func, null, result, p); }
    }
}
