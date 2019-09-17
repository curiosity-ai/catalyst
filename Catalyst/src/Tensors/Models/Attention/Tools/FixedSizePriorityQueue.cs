using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Catalyst.Tensors.Models.Tools
{
    public class FixedSizePriorityQueue<T> : IEnumerable<T>, IEnumerable
    {
        protected T[] m_rgtBottom;
        protected T[] m_rgtTop;
        protected int m_iCount;
        protected int m_iCapacity;
        protected int m_iBottomSize;
        protected int m_iTopSize;
        protected IComparer<T> comparer;

        public FixedSizePriorityQueue(int capacity)
        {
            if (capacity < 1)
                throw new Exception("priority queue capacity must be at least one!");
            m_iCount = 0;
            m_iCapacity = capacity;
            m_iBottomSize = m_iTopSize = 0;
            int length = Math.Max(capacity / 2, 1);
            m_rgtTop = new T[Math.Max(capacity - length, 1)];
            m_rgtBottom = new T[length];
        }

        public FixedSizePriorityQueue(int cap, IComparer<T> comp) : this(cap)
        {
            comparer = comp;
        }

        public int Count
        {
            get
            {
                return m_iCount;
            }
        }

        public int Capacity
        {
            get
            {
                return m_iCapacity;
            }
        }

        public void Clear()
        {
            m_iCount = 0;
            m_rgtBottom.Initialize();
            m_rgtTop.Initialize();
            m_iTopSize = m_iBottomSize = 0;
        }

        public bool Enqueue(T t)
        {
            if (m_iCapacity == m_iCount)
            {
                if (m_iCapacity == 1)
                {
                    if (!Better(t, m_rgtTop[0], true))
                        return false;
                    m_rgtTop[0] = t;
                    return true;
                }
                if (!Better(t, m_rgtBottom[0], true))
                    return false;
                m_rgtBottom[0] = t;
                int num = DownHeapify(m_rgtBottom, 0, m_iBottomSize, false);
                if (!SemiLeaf(num, m_iBottomSize))
                    return true;
                int i = CheckBoundaryUpwards(num);
                if (i == -1)
                    return true;
                UpHeapify(m_rgtTop, i, m_iTopSize, true);
                return true;
            }
            ++m_iCount;
            if (m_iBottomSize < m_iTopSize)
            {
                int index = m_iBottomSize++;
                m_rgtBottom[index] = t;
                int i = CheckBoundaryUpwards(index);
                if (i == -1)
                    UpHeapify(m_rgtBottom, index, m_iBottomSize, false);
                else
                    UpHeapify(m_rgtTop, i, m_iTopSize, true);
                return true;
            }
            int index1 = m_iTopSize++;
            m_rgtTop[index1] = t;
            int i1 = CheckBoundaryDownwards(index1);
            if (i1 == -1)
                UpHeapify(m_rgtTop, index1, m_iTopSize, true);
            else
                UpHeapify(m_rgtBottom, i1, m_iBottomSize, false);
            return true;
        }

        private int Parent(int i)
        {
            return (i - 1) / 2;
        }

        private int Left(int i)
        {
            return 2 * i + 1;
        }

        private int Right(int i)
        {
            return 2 * i + 2;
        }

        private bool IsLeft(int i)
        {
            return i % 2 == 1;
        }

        private bool IsRight(int i)
        {
            return i % 2 == 0;
        }

        private bool Leaf(int i, int iSize)
        {
            return Left(i) >= iSize;
        }

        private bool SemiLeaf(int i, int iSize)
        {
            return Right(i) >= iSize;
        }

        private int BottomNode(int i)
        {
            if (i < m_iBottomSize)
                return i;
            if (i % 2 == 1)
                return Parent(i);
            return i - 1;
        }

        private int TopNode1(int i)
        {
            if (Left(i) >= m_iBottomSize && Left(i) < m_iTopSize)
                return Left(i);
            return i;
        }

        private int TopNode2(int i)
        {
            if (i == m_iBottomSize - 1 && 1 == i % 2 && m_iTopSize > m_iBottomSize)
                return i + 1;
            if (Left(i) >= m_iBottomSize && Left(i) < m_iTopSize)
                return Left(i);
            return i;
        }

        private int UpHeapify(T[] rgt, int i, int iSize, bool fTop)
        {
            int i2;
            for (; i > 0; i = i2)
            {
                i2 = Parent(i);
                if (!Better(rgt[i], rgt[i2], fTop))
                    return i;
                Swap(rgt, i, rgt, i2);
            }
            return i;
        }

        private int DownHeapify(T[] rgt, int i, int iSize, bool fTop)
        {
            while (true)
            {
                int index1 = Left(i);
                int index2 = Right(i);
                int i2 = i;
                if (index1 < iSize && Better(rgt[index1], rgt[i2], fTop))
                    i2 = index1;
                if (index2 < iSize && Better(rgt[index2], rgt[i2], fTop))
                    i2 = index2;
                if (i2 != i)
                {
                    Swap(rgt, i, rgt, i2);
                    i = i2;
                }
                else
                    break;
            }
            return i;
        }

        private int CheckBoundaryUpwards(int iBottomPos)
        {
            int index1 = TopNode1(iBottomPos);
            int index2 = TopNode2(iBottomPos);
            int i1 = -1;
            if (Better(m_rgtBottom[iBottomPos], m_rgtTop[index1], true))
                i1 = index1;
            if (Better(m_rgtBottom[iBottomPos], m_rgtTop[index2], true) && (i1 == -1 || Better(m_rgtTop[index1], m_rgtTop[index2], true)))
                i1 = index2;
            if (i1 == -1)
                return -1;
            Swap(m_rgtTop, i1, m_rgtBottom, iBottomPos);
            return i1;
        }

        private int CheckBoundaryDownwards(int iTopPos)
        {
            int i2 = BottomNode(iTopPos);
            if (i2 == -1 || i2 >= m_iBottomSize || !Better(m_rgtBottom[i2], m_rgtTop[iTopPos], true))
                return -1;
            Swap(m_rgtTop, iTopPos, m_rgtBottom, i2);
            return i2;
        }

        private void Swap(T[] rgt1, int i1, T[] rgt2, int i2)
        {
            T obj = rgt1[i1];
            rgt1[i1] = rgt2[i2];
            rgt2[i2] = obj;
        }

        protected bool Better(T t1, T t2, bool fTop)
        {
            int num = comparer == null ? ((IComparable<T>)(object)t1).CompareTo(t2) : comparer.Compare(t1, t2);
            if (fTop)
                return num > 0;
            return num < 0;
        }

        public IEnumerator<T> GetEnumerator()
        {
            for (int i = 0; i < m_iTopSize; ++i)
                yield return m_rgtTop[i];
            for (int i = m_iBottomSize - 1; i >= 0; --i)
                yield return m_rgtBottom[i];
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
