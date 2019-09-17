using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Cpu;

namespace Catalyst.Tensors.CUDA
{
    /// <summary>
    /// Basic implementation of random ops for CUDA. All we do here is generate the tensors on the
    /// CPU then copy to the CUDA buffer. This is definitely not an optimal implementation.
    /// </summary>
    [OpsClass]
    public class CudaRandom
    {
        private readonly CpuAllocator cpuAllocator;
        private readonly CpuRandom cpuRandom;

        public CudaRandom()
        {
            this.cpuAllocator = new CpuAllocator();
            this.cpuRandom = new CpuRandom();
        }


        [RegisterOpStorageType("random_uniform", typeof(CudaStorage))]
        public void Uniform(Tensor result, int? seed, float min, float max)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Uniform(cpuCopy, seed, min, max);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_normal", typeof(CudaStorage))]
        public void Normal(Tensor result, int? seed, float mean, float stdv)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Normal(cpuCopy, seed, mean, stdv);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_exponential", typeof(CudaStorage))]
        public void Exponential(Tensor result, int? seed, float lambda)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Exponential(cpuCopy, seed, lambda);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_cauchy", typeof(CudaStorage))]
        public void Cauchy(Tensor result, int? seed, float median, float sigma)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Cauchy(cpuCopy, seed, median, sigma);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_lognormal", typeof(CudaStorage))]
        public void LogNormal(Tensor result, int? seed, float mean, float stdv)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.LogNormal(cpuCopy, seed, mean, stdv);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_geometric", typeof(CudaStorage))]
        public void Geometric(Tensor result, int? seed, float p)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Geometric(cpuCopy, seed, p);
                Ops.Copy(result, cpuCopy);
            }
        }

        [RegisterOpStorageType("random_bernoulli", typeof(CudaStorage))]
        public void Bernoulli(Tensor result, int? seed, float p)
        {
            using (var cpuCopy = new Tensor(cpuAllocator, result.ElementType, result.Sizes))
            {
                cpuRandom.Bernoulli(cpuCopy, seed, p);
                Ops.Copy(result, cpuCopy);
            }
        }
    }
}
