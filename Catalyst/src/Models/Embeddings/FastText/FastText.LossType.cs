namespace Catalyst.Models
{
    public partial class FastText
    {
        public enum LossType
        {
            HierarchicalSoftMax,
            SoftMax,
            NegativeSampling,
            OneVsAll
        }
    }
}