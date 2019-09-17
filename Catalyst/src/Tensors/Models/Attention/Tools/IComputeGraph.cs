using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors
{
    public interface IComputeGraph
    {
        IWeightMatrix MulBatch(IWeightMatrix m1, IWeightMatrix m2, int batchSize);

        IWeightMatrix Mul(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2);
        IWeightMatrix Tanh(IWeightMatrix w, bool updateWeightsInPlace = false);
        IWeightMatrix Sigmoid(IWeightMatrix w, bool updateWeightsInPlace = false);

        IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3);

        IWeightMatrix EltMulMulAdd(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4);

        List<IWeightMatrix> UnFolderRow(IWeightMatrix m, int n, bool gradient = true);
        IWeightMatrix PermuteBatch(IWeightMatrix m, int batchSize);

        IWeightMatrix View(IWeightMatrix m, int r, int c);

        IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2);

        IWeightMatrix ConcatColumns(IWeightMatrix m1, IWeightMatrix m2);

        void Backward();
        void RunTopBackward();

        IWeightMatrix PeekRow(IWeightMatrix w, int ix, int num = 1);
        IWeightMatrix Dropout(IWeightMatrix V, float drop_prob);

        IWeightMatrix Softmax(IWeightMatrix w, bool bp = true);

        IWeightMatrix ConcatColumns(params IWeightMatrix[] wl);        

        List<IWeightMatrix> SplitColumns2(IWeightMatrix w, params int[] sizes);
        (IWeightMatrix r1, IWeightMatrix r2) SplitColumns(IWeightMatrix w, int size1, int size2);
        (IWeightMatrix r1, IWeightMatrix r2, IWeightMatrix r3) SplitColumns(IWeightMatrix w, int size1, int size2, int size3);

        IWeightMatrix ConcatRows(List<IWeightMatrix> wl);

        IWeightMatrix RepeatRows(IWeightMatrix w, int n);

        IWeightMatrix Transpose2(IWeightMatrix w);

        IWeightMatrix ConcatRowColumn(List<IWeightMatrix> wl1, List<IWeightMatrix> wl2);
		
		 IWeightMatrix Mul(IWeightMatrix w, float v);

        IWeightMatrix LayerNorm(IWeightMatrix src, IWeightMatrix alpha, IWeightMatrix beta, float eps = 1e-09f);
    }
}
