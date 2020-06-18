using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Catalyst.Models
{
    public partial class FastText
    {
        public class ThreadState //Per thread
        {
            public readonly int ThreadID;
            public long NumberOfExamples;
            public float Loss;
            public readonly float[] Hidden;
            public readonly float[] Output;
            public readonly float[] Gradient;
            public readonly float[] t_log;
            public readonly float[] t_sigmoid;
            public readonly Line[] Corpus;
            public readonly CancellationToken CancellationToken;
            public readonly TrainingHistory TrainingHistory;
            public int NegativePosition;


            public ThreadState(Line[] corpus, int hlen, int olen, int glen, int thread, CancellationToken token)
            {
                t_log     = new float[Utils.LOG_TABLE_SIZE];
                t_sigmoid = new float[Utils.SIGMOID_TABLE_SIZE];
                
                Utils.init(ref t_log, ref t_sigmoid);

                Loss              = 0f;
                NumberOfExamples  = 1;
                Corpus            = corpus;
                NegativePosition  = 0;
                Hidden            = new float[hlen];
                Output            = new float[olen];
                Gradient          = new float[glen];
                ThreadID          = thread;
                CancellationToken = token;

                if (thread == 0)
                {
                    TrainingHistory = new TrainingHistory();
                }
                else
                {
                    TrainingHistory = null;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float Log(float x)
            {
                if (x >= 0.99f) 
                { 
                    return 0.0f; 
                }
                else
                {
                    int i = (int)(x * Utils.LOG_TABLE_SIZEf);
                    return t_log[i];
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float Sigmoid(float x)
            {
                if (x <= -(Utils.MAX_SIGMOIDf - 0.1f)) 
                { 
                    return 0.0f; 
                }
                else if (x >= (Utils.MAX_SIGMOIDf - 0.1f)) 
                {
                    return 1.0f; 
                }
                else
                {
                    int i = (int)((x + Utils.MAX_SIGMOIDf) * Utils.SIGMOID_TABLE_SIZEf / Utils.MAX_SIGMOIDf / 2.0f);
                    return t_sigmoid[i];
                }
            }

            public float GetLoss()
            {
                return Loss / NumberOfExamples;
            }
        }

        public static class Utils
        {
            public const byte spaceCode = (byte)' ';
            public const int SIGMOID_TABLE_SIZE = 512;
            public const int MAX_SIGMOID = 8;
            public const int LOG_TABLE_SIZE = 512;

            public const float SIGMOID_TABLE_SIZEf = (float)SIGMOID_TABLE_SIZE;
            public const float MAX_SIGMOIDf = (float)MAX_SIGMOID;
            public const float LOG_TABLE_SIZEf = (float)LOG_TABLE_SIZE;

            public static void init(ref float[] t_log, ref float[] t_sigmoid)
            {
                for (int i = 0; i < LOG_TABLE_SIZE; i++)
                {
                    var x = ((float)i + 1e-5f) / LOG_TABLE_SIZEf;
                    t_log[i] = (float)Math.Log(x);
                }

                for (int i = 0; i < SIGMOID_TABLE_SIZE; i++)
                {
                    double x = (double)(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZEf - MAX_SIGMOIDf;
                    t_sigmoid[i] = (float)(1.0f / (1.0f + Math.Exp(-x)));
                }
            }
        }
    }
}