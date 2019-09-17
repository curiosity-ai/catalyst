using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{

    public class ComparableItem<T>
    {
        public float Score { get; }
        public T Value { get; }

        public ComparableItem(float score, T value)
        {
            Score = score;
            Value = value;
        }
    }

    public class ComparableItemComparer<T> : IComparer<ComparableItem<T>>
    {
        public ComparableItemComparer(bool fAscending)
        {
            m_fAscending = fAscending;
        }

        public int Compare(ComparableItem<T> x, ComparableItem<T> y)
        {
            int iSign = Math.Sign(x.Score - y.Score);
            if (!m_fAscending)
                iSign = -iSign;
            return iSign;
        }

        protected bool m_fAscending;
    }
}
