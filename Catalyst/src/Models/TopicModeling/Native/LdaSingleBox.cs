using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Catalyst.Models.Native
{

    /*  Code derived from original ML.NET implementaion at https://github.com/dotnet/machinelearning, 
     *  using LightLDA implementation from https://github.com/microsoft/LightLDA
     *  Both licensed unde the MIT License (MIT)
     *  LightLDA - Copyright (c) Microsoft Corporation
     *  ML.NET   - Copyright (c) 2018 .NET Foundation
     */

    internal sealed class LdaSingleBox : IDisposable
    {
        private LdaInterface.LdaEngine _engine;
        private bool _isDisposed;
        private int[] _topics;
        private int[] _probabilities;
        private int[] _summaryTerm;
        private float[] _summaryTermProb;
        private readonly int _likelihoodInterval;
        private readonly float _alpha;
        private readonly float _beta;
        private readonly int _mhStep;
        private readonly int _numThread;
        private readonly int _numSummaryTerms;
        private readonly bool _denseOutput;

        internal readonly int NumTopic;
        internal readonly int NumVocab;
        internal LdaSingleBox(int numTopic, int numVocab, float alpha, float beta, int numIter, int likelihoodInterval, int numThread, int mhstep, int numSummaryTerms, bool denseOutput, int maxDocToken)
        {
            NumTopic = numTopic;
            NumVocab = numVocab;
            _alpha = alpha;
            _beta = beta;
            _mhStep = mhstep;
            _numSummaryTerms = numSummaryTerms;
            _denseOutput = denseOutput;
            _likelihoodInterval = likelihoodInterval;
            _numThread = numThread;

            _topics = new int[numTopic];
            _probabilities = new int[numTopic];

            _summaryTerm = new int[_numSummaryTerms];
            _summaryTermProb = new float[_numSummaryTerms];

            _engine = LdaInterface.CreateEngine(numTopic, numVocab, alpha, beta, numIter, likelihoodInterval, numThread, mhstep, maxDocToken);
        }

        internal void AllocateModelMemory(int numTopic, int numVocab, long tableSize, long aliasTableSize)
        {
            Debug.Assert(numTopic >= 0);
            Debug.Assert(numVocab >= 0);
            Debug.Assert(tableSize >= 0);
            Debug.Assert(aliasTableSize >= 0);
            LdaInterface.AllocateModelMemory(_engine, numVocab, numTopic, tableSize, aliasTableSize);
        }

        internal void AllocateDataMemory(int docNum, long corpusSize)
        {
            Debug.Assert(docNum >= 0);
            Debug.Assert(corpusSize >= 0);
            LdaInterface.AllocateDataMemory(_engine, docNum, corpusSize);
        }

        internal void Train(string trainOutput)
        {
            if (string.IsNullOrWhiteSpace(trainOutput))
                LdaInterface.Train(_engine, null);
            else
                LdaInterface.Train(_engine, trainOutput);
        }

        internal void GetModelStat(out long memBlockSize, out long aliasMemBlockSize)
        {
            LdaInterface.GetModelStat(_engine, out memBlockSize, out aliasMemBlockSize);
        }

        internal void Test(int numBurninIter, float[] logLikelihood)
        {
            Debug.Assert(numBurninIter >= 0);
            var pLogLikelihood = new float[numBurninIter];
            LdaInterface.Test(_engine, numBurninIter, pLogLikelihood);
            logLikelihood = pLogLikelihood.Select(item => (float)item).ToArray();
        }

        internal void CleanData()
        {
            LdaInterface.CleanData(_engine);
        }

        internal void CleanModel()
        {
            LdaInterface.CleanModel(_engine);
        }

        internal void CopyModel(LdaSingleBox trainer, int wordId)
        {
            int length = NumTopic;
            LdaInterface.GetWordTopic(trainer._engine, wordId, _topics, _probabilities, ref length);
            LdaInterface.SetWordTopic(_engine, wordId, _topics, _probabilities, length);
        }

        internal void SetAlphaSum(float averageDocLength)
        {
            LdaInterface.SetAlphaSum(_engine, averageDocLength);
        }

        internal int LoadDoc(ReadOnlySpan<int> termID, ReadOnlySpan<double> termVal, int termNum, int numVocab)
        {
            Debug.Assert(numVocab == NumVocab);
            Debug.Assert(termNum > 0);
            Debug.Assert(termID.Length >= termNum);
            Debug.Assert(termVal.Length >= termNum);

            int[] pID = new int[termNum];
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            termID.Slice(0, termNum).CopyTo(pID);
            return LdaInterface.FeedInData(_engine, pID, pVal, termNum, NumVocab);
        }

        internal List<KeyValuePair<int, float>> GetDocTopicVector(int docID)
        {
            int numTopicReturn = NumTopic;
            LdaInterface.GetDocTopic(_engine, docID, _topics, _probabilities, ref numTopicReturn);
            var topicRet = new List<KeyValuePair<int, float>>();
            int currentTopic = 0;
            for (int i = 0; i < numTopicReturn; i++)
            {
                if (_denseOutput)
                {
                    while (currentTopic < _topics[i])
                    {
                        //use a value to smooth the count so that we get dense output on each topic
                        //the smooth value is usually set to 0.1
                        topicRet.Add(new KeyValuePair<int, float>(currentTopic, (float)_alpha));
                        currentTopic++;
                    }
                    topicRet.Add(new KeyValuePair<int, float>(_topics[i], _probabilities[i] + (float)_alpha));
                    currentTopic++;
                }
                else
                {
                    topicRet.Add(new KeyValuePair<int, float>(_topics[i], (float)_probabilities[i]));
                }
            }

            if (_denseOutput)
            {
                while (currentTopic < NumTopic)
                {
                    topicRet.Add(new KeyValuePair<int, float>(currentTopic, (float)_alpha));
                    currentTopic++;
                }
            }
            return topicRet;
        }

        internal List<KeyValuePair<int, float>> TestDoc(ReadOnlySpan<int> termID, ReadOnlySpan<double> termVal, int termNum, int numBurninIter, bool reset)
        {
            Debug.Assert(termNum > 0);
            Debug.Assert(termVal.Length >= termNum);
            Debug.Assert(termID.Length >= termNum);

            int[] pID = new int[termNum];
            int[] pVal = new int[termVal.Length];
            for (int i = 0; i < termVal.Length; i++)
                pVal[i] = (int)termVal[i];
            int[] pTopic = new int[NumTopic];
            int[] pProb = new int[NumTopic];
            termID.Slice(0, termNum).CopyTo(pID);

            int numTopicReturn = NumTopic;

            LdaInterface.TestOneDoc(_engine, pID, pVal, termNum, pTopic, pProb, ref numTopicReturn, numBurninIter, reset);

            // PREfast suspects that the value of numTopicReturn could be changed in _engine->TestOneDoc, which might result in read overrun in the following loop.
            if (numTopicReturn > NumTopic)
            {
                Debug.Assert(false);
                numTopicReturn = NumTopic;
            }

            var topicRet = new List<KeyValuePair<int, float>>();
            for (int i = 0; i < numTopicReturn; i++)
                topicRet.Add(new KeyValuePair<int, float>(pTopic[i], (float)pProb[i]));
            return topicRet;
        }

        internal void InitializeBeforeTrain()
        {
            LdaInterface.InitializeBeforeTrain(_engine);
        }

        internal void InitializeBeforeTest()
        {
            LdaInterface.InitializeBeforeTest(_engine);
        }

        internal KeyValuePair<int, int>[] GetModel(int wordId)
        {
            int length = NumTopic;
            LdaInterface.GetWordTopic(_engine, wordId, _topics, _probabilities, ref length);
            var wordTopicVector = new KeyValuePair<int, int>[length];

            for (int i = 0; i < length; i++)
                wordTopicVector[i] = new KeyValuePair<int, int>(_topics[i], _probabilities[i]);
            return wordTopicVector;
        }

        internal KeyValuePair<int, float>[] GetTopicSummary(int topicId)
        {
            int length = _numSummaryTerms;
            LdaInterface.GetTopicSummary(_engine, topicId, _summaryTerm, _summaryTermProb, ref length);
            var topicSummary = new KeyValuePair<int, float>[length];

            for (int i = 0; i < length; i++)
                topicSummary[i] = new KeyValuePair<int, float>(_summaryTerm[i], _summaryTermProb[i]);
            return topicSummary;
        }

        internal void SetModel(int termID, int[] topicID, int[] topicProb, int topicNum)
        {
            Debug.Assert(termID >= 0);
            Debug.Assert(topicNum <= NumTopic);
            Array.Copy(topicID, _topics, topicNum);
            Array.Copy(topicProb, _probabilities, topicNum);
            LdaInterface.SetWordTopic(_engine, termID, _topics, _probabilities, topicNum);
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;
            _isDisposed = true;
            LdaInterface.DestroyEngine(_engine);
            _engine.Ptr = IntPtr.Zero;
        }
    }
}
