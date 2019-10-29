using System;
using System.Security;
using System.Runtime.InteropServices;

namespace Catalyst.Models.Native
{
    /*  Code derived from original ML.NET implementaion at https://github.com/dotnet/machinelearning, 
     *  using LightLDA implementation from https://github.com/microsoft/LightLDA
     *  Both licensed unde the MIT License (MIT)
     *  LightLDA - Copyright (c) Microsoft Corporation
     *  ML.NET   - Copyright (c) 2018 .NET Foundation
     */
    internal static class LdaInterface
    {
        public struct LdaEngine
        {
            public IntPtr Ptr;
        }

        private const string NativePath = "LdaNative";
        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern LdaEngine CreateEngine(int numTopic, int numVocab, float alphaSum, float beta, int numIter,
            int likelihoodInterval, int numThread, int mhstep, int maxDocToken);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void AllocateModelMemory(LdaEngine engine, int numTopic, int numVocab, long tableSize, long aliasTableSize);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void AllocateDataMemory(LdaEngine engine, int docNum, long corpusSize);

        [DllImport(NativePath, CharSet = CharSet.Ansi), SuppressUnmanagedCodeSecurity]
        internal static extern void Train(LdaEngine engine, string trainOutput);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetModelStat(LdaEngine engine, out long memBlockSize, out long aliasMemBlockSize);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void Test(LdaEngine engine, int numBurninIter, float[] pLogLikelihood);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void CleanData(LdaEngine engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void CleanModel(LdaEngine engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void DestroyEngine(LdaEngine engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetWordTopic(LdaEngine engine, int wordId, int[] pTopic, int[] pProb, ref int length);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void SetWordTopic(LdaEngine engine, int wordId, int[] pTopic, int[] pProb, int length);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void SetAlphaSum(LdaEngine engine, float avgDocLength);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern int FeedInData(LdaEngine engine, int[] termId, int[] termFreq, int termNum, int numVocab);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern int FeedInDataDense(LdaEngine engine, int[] termFreq, int termNum, int numVocab);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetDocTopic(LdaEngine engine, int docId, int[] pTopic, int[] pProb, ref int numTopicReturn);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void GetTopicSummary(LdaEngine engine, int topicId, int[] pWords, float[] pProb, ref int numTopicReturn);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void TestOneDoc(LdaEngine engine, int[] termId, int[] termFreq, int termNum, int[] pTopics, int[] pProbs, ref int numTopicsMax, int numBurnIter, bool reset);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void TestOneDocDense(LdaEngine engine, int[] termFreq, int termNum, int[] pTopics, int[] pProbs, ref int numTopicsMax, int numBurninIter, bool reset);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void InitializeBeforeTrain(LdaEngine engine);

        [DllImport(NativePath), SuppressUnmanagedCodeSecurity]
        internal static extern void InitializeBeforeTest(LdaEngine engine);
    }
}
