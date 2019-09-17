using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public interface IWeightMatrix : IDisposable
    {
        int Rows { get; set; }
        int Columns { get; set; }

        int DeviceId { get; set; }

        Dictionary<int, int> RowToBeUpdated { get; set; }

        void CleanCache();

        float GetWeightAt(int offset);
        void SetWeightAt(float val, int offset);
        void SetGradientAt(float val, int offset);

        void SetGradientByWeight(IWeightMatrix src);

        void SetWeightAtRow(int row, float[] val);

        float[] ToWeightArray();
        int GetMaxWeightIdx();

        List<int> GetTopNMaxWeightIdx(int topN);

        void SetWeightArray(float[] v);

        void ReleaseWeight();

        void ClearGradient();
        void ClearWeight();

        void Save(Stream stream);
        void Load(Stream stream);

        void CopyWeights(IWeightMatrix src);
        void AddGradient(IWeightMatrix src);
    }
}
