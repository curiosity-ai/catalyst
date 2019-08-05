using System;
using System.Collections.Generic;

namespace Catalyst.Models
{
    public class ModelTranslation
    {
        private Matrix TranslationMatrix;
        public Dictionary<string, string> Dictionary = new Dictionary<string, string>();

        public void ComputeAlignment(FastText source, FastText target)
        {
            //find words that exist on both models
            int dim = source.Data.Dimensions;
            if (target.Data.Dimensions != dim) { throw new Exception("Source and Target models must have the same dimensions!"); }
            int N = 10_000;
            var dict = TranslationDictionary.GetDictionary(source.Language, target.Language, N * 3);

            var A = new Matrix(dict.Count, dim);
            var B = new Matrix(dict.Count, dim);
            int k = 0;
            foreach (var kv in dict)
            {
                if ((source.GetWordIndex(kv.Key) > -1) && (target.GetWordIndex(kv.Value) > -1))
                {
                    Dictionary.Add(kv.Key, kv.Value);

                    A.Data[k] = source.GetVector(kv.Key, source.Language);
                    B.Data[k] = target.GetVector(kv.Value, target.Language);
                    k++;
                }
                if (k == N) { break; }
            }

            A.ResizeAndFillRows(k, 0);
            B.ResizeAndFillRows(k, 0);

            var U = B.Transpose().Multiply(A);

            CalculateSVD(ref U.Data, out float[] w, out float[][] v);

            TranslationMatrix = U.Multiply(new Matrix(v).Transpose());

            //M = A*B
            //U,V = SVD(M)
            //W = U dot V'
        }

        public float[] Translate(float[] vector)
        {
            var v = new float[vector.Length];
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = TranslationMatrix.DotRow(ref vector, i);
            }
            return v;
        }

        // Just a copy-paste of SVD algorithm from Numerical Recipes but updated for C#
        // (as authors state, the code is aimed to be machine readable, so blame them
        // for all those c/f/g/h/s variable)
        // A = U dot W dot V' , using U = A for input
        private static void CalculateSVD(ref float[][] u, out float[] w, out float[][] v_t)
        {
            // number of rows in A
            int m = u.Length;
            // number of columns in A
            int n = u[0].Length;

            if (m < n)
            {
                throw new ArgumentException("Number of rows in A must be greater or equal to number of columns");
            }

            int flag, i, its, j, jj, k, l = 0, nm = 0;
            float anorm, c, f, g, h, s, scale, x, y, z;

            var rv1 = new float[n];

            w = new float[n];
            v_t = new float[n][];
            for (i = 0; i < n; i++)
            {
                v_t[i] = new float[n];
            }

            // householder reduction to bidiagonal form
            g = scale = anorm = 0.0f;

            for (i = 0; i < n; i++)
            {
                l = i + 1;
                rv1[i] = scale * g;
                g = s = scale = 0;

                if (i < m)
                {
                    for (k = i; k < m; k++)
                    {
                        scale += System.Math.Abs(u[k][i]);
                    }

                    if (scale != 0.0)
                    {
                        for (k = i; k < m; k++)
                        {
                            u[k][i] /= scale;
                            s += u[k][i] * u[k][i];
                        }

                        f = u[i][i];
                        g = -Sign(System.Math.Sqrt(s), f);
                        h = f * g - s;
                        u[i][i] = f - g;

                        if (i != n - 1)
                        {
                            for (j = l; j < n; j++)
                            {
                                for (s = 0.0f, k = i; k < m; k++)
                                {
                                    s += u[k][i] * u[k][j];
                                }

                                f = s / h;

                                for (k = i; k < m; k++)
                                {
                                    u[k][j] += f * u[k][i];
                                }
                            }
                        }

                        for (k = i; k < m; k++)
                        {
                            u[k][i] *= scale;
                        }
                    }
                }

                w[i] = scale * g;
                g = s = scale = 0;

                if ((i < m) && (i != n - 1))
                {
                    for (k = l; k < n; k++)
                    {
                        scale += System.Math.Abs(u[i][k]);
                    }

                    if (scale != 0.0)
                    {
                        for (k = l; k < n; k++)
                        {
                            u[i][k] /= scale;
                            s += u[i][k] * u[i][k];
                        }

                        f = u[i][l];
                        g = -Sign(System.Math.Sqrt(s), f);
                        h = f * g - s;
                        u[i][l] = f - g;

                        for (k = l; k < n; k++)
                        {
                            rv1[k] = u[i][k] / h;
                        }

                        if (i != m - 1)
                        {
                            for (j = l; j < m; j++)
                            {
                                for (s = 0, k = l; k < n; k++)
                                {
                                    s += u[j][k] * u[i][k];
                                }
                                for (k = l; k < n; k++)
                                {
                                    u[j][k] += s * rv1[k];
                                }
                            }
                        }

                        for (k = l; k < n; k++)
                        {
                            u[i][k] *= scale;
                        }
                    }
                }
                anorm = (float)Math.Max(anorm, (Math.Abs(w[i]) + Math.Abs(rv1[i])));
            }

            // accumulation of right-hand transformations
            for (i = n - 1; i >= 0; i--)
            {
                if (i < n - 1)
                {
                    if (g != 0.0)
                    {
                        for (j = l; j < n; j++)
                        {
                            v_t[j][i] = (u[i][j] / u[i][l]) / g;
                        }

                        for (j = l; j < n; j++)
                        {
                            for (s = 0, k = l; k < n; k++)
                            {
                                s += u[i][k] * v_t[k][j];
                            }
                            for (k = l; k < n; k++)
                            {
                                v_t[k][j] += s * v_t[k][i];
                            }
                        }
                    }
                    for (j = l; j < n; j++)
                    {
                        v_t[i][j] = v_t[j][i] = 0;
                    }
                }
                v_t[i][i] = 1;
                g = rv1[i];
                l = i;
            }

            // accumulation of left-hand transformations
            for (i = n - 1; i >= 0; i--)
            {
                l = i + 1;
                g = w[i];

                if (i < n - 1)
                {
                    for (j = l; j < n; j++)
                    {
                        u[i][j] = 0;
                    }
                }

                if (g != 0)
                {
                    g = 1f / g;

                    if (i != n - 1)
                    {
                        for (j = l; j < n; j++)
                        {
                            for (s = 0, k = l; k < m; k++)
                            {
                                s += u[k][i] * u[k][j];
                            }

                            f = (s / u[i][i]) * g;

                            for (k = i; k < m; k++)
                            {
                                u[k][j] += f * u[k][i];
                            }
                        }
                    }

                    for (j = i; j < m; j++)
                    {
                        u[j][i] *= g;
                    }
                }
                else
                {
                    for (j = i; j < m; j++)
                    {
                        u[j][i] = 0;
                    }
                }
                ++u[i][i];
            }

            // diagonalization of the bidiagonal form: Loop over singular values
            // and over allowed iterations
            for (k = n - 1; k >= 0; k--)
            {
                for (its = 1; its <= 30; its++)
                {
                    flag = 1;

                    for (l = k; l >= 0; l--)
                    {
                        // test for splitting
                        nm = l - 1;

                        if (System.Math.Abs(rv1[l]) + anorm == anorm)
                        {
                            flag = 0;
                            break;
                        }

                        if (System.Math.Abs(w[nm]) + anorm == anorm)
                            break;
                    }

                    if (flag != 0)
                    {
                        c = 0;
                        s = 1f;
                        for (i = l; i <= k; i++)
                        {
                            f = s * rv1[i];

                            if (Math.Abs(f) + anorm != anorm)
                            {
                                g = w[i];
                                h = Pythag(f, g);
                                w[i] = h;
                                h = 1f / h;
                                c = g * h;
                                s = -f * h;

                                for (j = 0; j < m; j++)
                                {
                                    y = u[j][nm];
                                    z = u[j][i];
                                    u[j][nm] = y * c + z * s;
                                    u[j][i] = z * c - y * s;
                                }
                            }
                        }
                    }

                    z = w[k];

                    if (l == k)
                    {
                        // convergence
                        if (z < 0.0)
                        {
                            // singular value is made nonnegative
                            w[k] = -z;

                            for (j = 0; j < n; j++)
                            {
                                v_t[j][k] = -v_t[j][k];
                            }
                        }
                        break;
                    }

                    if (its == 30)
                    {
                        throw new ApplicationException("No convergence in 30 svdcmp iterations");
                    }

                    // shift from bottom 2-by-2 minor
                    x = w[l];
                    nm = k - 1;
                    y = w[nm];
                    g = rv1[nm];
                    h = rv1[k];
                    f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2f * h * y);
                    g = Pythag(f, 1f);
                    f = ((x - z) * (x + z) + h * ((y / (f + Sign(g, f))) - h)) / x;

                    // next QR transformation
                    c = s = 1f;

                    for (j = l; j <= nm; j++)
                    {
                        i = j + 1;
                        g = rv1[i];
                        y = w[i];
                        h = s * g;
                        g = c * g;
                        z = Pythag(f, h);
                        rv1[j] = z;
                        c = f / z;
                        s = h / z;
                        f = x * c + g * s;
                        g = g * c - x * s;
                        h = y * s;
                        y *= c;

                        for (jj = 0; jj < n; jj++)
                        {
                            x = v_t[jj][j];
                            z = v_t[jj][i];
                            v_t[jj][j] = x * c + z * s;
                            v_t[jj][i] = z * c - x * s;
                        }

                        z = Pythag(f, h);
                        w[j] = z;

                        if (z != 0)
                        {
                            z = 1f / z;
                            c = f * z;
                            s = h * z;
                        }

                        f = c * g + s * y;
                        x = c * y - s * g;

                        for (jj = 0; jj < m; jj++)
                        {
                            y = u[jj][j];
                            z = u[jj][i];
                            u[jj][j] = y * c + z * s;
                            u[jj][i] = z * c - y * s;
                        }
                    }

                    rv1[l] = 0f;
                    rv1[k] = f;
                    w[k] = x;
                }
            }
        }

        private static float Sign(double a, float b)
        {
            return (b >= 0.0) ? (float)Math.Abs(a) : (float)-Math.Abs(a);
        }

        private static float Pythag(float a, float b)
        {
            double at = Math.Abs(a), bt = Math.Abs(b), ct, result;

            if (at > bt)
            {
                ct = bt / at;
                result = at * Math.Sqrt(1.0 + ct * ct);
            }
            else if (bt > 0.0)
            {
                ct = at / bt;
                result = bt * Math.Sqrt(1.0 + ct * ct);
            }
            else
            {
                result = 0.0;
            }

            return (float)result;
        }
    }
}