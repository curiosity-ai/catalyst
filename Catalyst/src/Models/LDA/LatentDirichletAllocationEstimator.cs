using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Collections.Immutable;
using System.Linq;

namespace Catalyst.Models
{
    /// <summary>
    /// The LDA transform implements <a href="https://arxiv.org/abs/1412.1576">LightLDA</a>, a state-of-the-art implementation of Latent Dirichlet Allocation.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    ///
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | Vector of <xref:System.Single> |
    /// | Output column data type | Vector of <xref:System.Single>|
    ///
    ///  Latent Dirichlet Allocation is a well-known [topic modeling](https://en.wikipedia.org/wiki/Topic_model) algorithm that infers semantic structure from text data,
    ///  and ultimately helps answer the question on "what is this document about?".
    ///  It can be used to featurize any text fields as low-dimensional topical vectors.
    ///  LightLDA is an extremely efficient implementation of LDA that incorporates a number of
    ///  optimization techniques.
    ///  With the LDA transform, ML.NET users can train a topic model to produce 1 million topics with 1 million words vocabulary
    ///  on a 1-billion-token document set one a single machine in a few hours(typically, LDA at this scale takes days and requires large clusters).
    ///  The most significant innovation is a super-efficient $O(1)$. [Metropolis-Hastings sampling algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm),
    ///  whose running cost is agnostic of model size, allowing it to converges nearly an order of magnitude faster than other [Gibbs samplers](https://en.wikipedia.org/wiki/Gibbs_sampling).
    ///
    ///  In an ML.NET pipeline, this estimator requires the output of some preprocessing, as its input.
    ///  A typical pipeline operating on text would require text normalization, tokenization and producing n-grams to supply to the LDA estimator.
    ///  See the example usage in the See Also section for usage suggestions.
    ///
    ///  If we have the following three examples of text, as data points, and use the LDA transform with the number of topics set to 3,
    ///  we would get the results displayed in the table below. Example documents:
    ///  * I like to eat bananas.
    ///  * I eat bananas everyday.
    ///  * First celebrated in 1970, Earth Day now includes events in more than 193 countries,
    ///    which are now coordinated globally by the Earth Day Network.
    ///
    ///  Notice the similarity in values of the first and second row, compared to the third,
    ///  and see how those values are indicative of similarities between those two (small) bodies of text.
    ///
    ///  | Topic1  | Topic2  | Topic 3 |
    ///  | ------- | ------- | ------- |
    ///  |  0.5714 | 0.0000  | 0.4286  |
    ///  |  0.5714 | 0.0000  | 0.4286  |
    ///  |  0.2400 | 0.3200  | 0.4400  |
    ///
    ///  For more technical details you can consult the following papers.
    ///  * [LightLDA: Big Topic Models on Modest Computer Clusters](https://arxiv.org/abs/1412.1576)
    ///  * [LightLDA](https://github.com/Microsoft/LightLDA)
    ///
    /// Check the See Also section for links to usage examples.
    /// ]]></format>
    /// </remarks>
    /// <seealso cref="TextCatalog.LatentDirichletAllocation(TransformsCatalog.TextTransforms, string, string, int, float, float, int, int, int, int, int, int, int, bool)"/>
    public sealed class LatentDirichletAllocationEstimator
    {
        internal static class Defaults
        {
            public const int NumberOfTopics = 100;
            public const float AlphaSum = 100;
            public const float Beta = 0.01f;
            public const int SamplingStepCount = 4;
            public const int MaximumNumberOfIterations = 200;
            public const int LikelihoodInterval = 5;
            public const int NumberOfThreads = 0;
            public const int MaximumTokenCountPerDocument = 512;
            public const int NumberOfSummaryTermsPerTopic = 10;
            public const int NumberOfBurninIterations = 10;
            public const bool ResetRandomGenerator = false;
        }

        private readonly IHost _host;
        private readonly ImmutableArray<ColumnOptions> _columns;

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="inputColumnName">Name of the column to transform. If set to <see langword="null"/>, the value of the <paramref name="outputColumnName"/> will be used as source.</param>
        /// <param name="numberOfTopics">The number of topics.</param>
        /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
        /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
        /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
        /// <param name="maximumNumberOfIterations">Number of iterations.</param>
        /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
        /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
        /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
        /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
        /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
        /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
        internal LatentDirichletAllocationEstimator(IHostEnvironment env,
            string outputColumnName, string inputColumnName = null,
            int numberOfTopics = Defaults.NumberOfTopics,
            float alphaSum = Defaults.AlphaSum,
            float beta = Defaults.Beta,
            int samplingStepCount = Defaults.SamplingStepCount,
            int maximumNumberOfIterations = Defaults.MaximumNumberOfIterations,
            int numberOfThreads = Defaults.NumberOfThreads,
            int maximumTokenCountPerDocument = Defaults.MaximumTokenCountPerDocument,
            int numberOfSummaryTermsPerTopic = Defaults.NumberOfSummaryTermsPerTopic,
            int likelihoodInterval = Defaults.LikelihoodInterval,
            int numberOfBurninIterations = Defaults.NumberOfBurninIterations,
            bool resetRandomGenerator = Defaults.ResetRandomGenerator)
            : this(env, new[] { new ColumnOptions(outputColumnName, inputColumnName ?? outputColumnName,
                numberOfTopics, alphaSum, beta, samplingStepCount, maximumNumberOfIterations, likelihoodInterval, numberOfThreads, maximumTokenCountPerDocument,
                numberOfSummaryTermsPerTopic, numberOfBurninIterations, resetRandomGenerator) })
        { }

        /// <include file='doc.xml' path='doc/members/member[@name="LightLDA"]/*' />
        /// <param name="env">The environment.</param>
        /// <param name="columns">Describes the parameters of the LDA process for each column pair.</param>
        internal LatentDirichletAllocationEstimator(IHostEnvironment env, params ColumnOptions[] columns)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(LatentDirichletAllocationEstimator));
            _columns = columns.ToImmutableArray();
        }

        /// <summary>
        /// Describes how the transformer handles one column pair.
        /// </summary>
        internal sealed class ColumnOptions
        {
            /// <summary>
            /// Name of the column resulting from the transformation of <cref see="InputColumnName"/>.
            /// </summary>
            public readonly string Name;
            /// <summary>
            /// Name of column to transform.
            /// </summary>
            public readonly string InputColumnName;
            /// <summary>
            /// The number of topics.
            /// </summary>
            public readonly int NumberOfTopics;
            /// <summary>
            /// Dirichlet prior on document-topic vectors.
            /// </summary>
            public readonly float AlphaSum;
            /// <summary>
            /// Dirichlet prior on vocab-topic vectors.
            /// </summary>
            public readonly float Beta;
            /// <summary>
            /// Number of Metropolis Hasting step.
            /// </summary>
            public readonly int SamplingStepCount;
            /// <summary>
            /// Number of iterations.
            /// </summary>
            public readonly int NumberOfIterations;
            /// <summary>
            /// Compute log likelihood over local dataset on this iteration interval.
            /// </summary>
            public readonly int LikelihoodInterval;
            /// <summary>
            /// The number of training threads.
            /// </summary>
            public readonly int NumberOfThreads;
            /// <summary>
            /// The threshold of maximum count of tokens per doc.
            /// </summary>
            public readonly int MaximumTokenCountPerDocument;
            /// <summary>
            /// The number of words to summarize the topic.
            /// </summary>
            public readonly int NumberOfSummaryTermsPerTopic;
            /// <summary>
            /// The number of burn-in iterations.
            /// </summary>
            public readonly int NumberOfBurninIterations;
            /// <summary>
            /// Reset the random number generator for each document.
            /// </summary>
            public readonly bool ResetRandomGenerator;

            /// <summary>
            /// Describes how the transformer handles one column pair.
            /// </summary>
            /// <param name="name">The column containing the output scores over a set of topics, represented as a vector of floats. </param>
            /// <param name="inputColumnName">The column representing the document as a vector of floats.A null value for the column means <paramref name="inputColumnName"/> is replaced. </param>
            /// <param name="numberOfTopics">The number of topics.</param>
            /// <param name="alphaSum">Dirichlet prior on document-topic vectors.</param>
            /// <param name="beta">Dirichlet prior on vocab-topic vectors.</param>
            /// <param name="samplingStepCount">Number of Metropolis Hasting step.</param>
            /// <param name="maximumNumberOfIterations">Number of iterations.</param>
            /// <param name="likelihoodInterval">Compute log likelihood over local dataset on this iteration interval.</param>
            /// <param name="numberOfThreads">The number of training threads. Default value depends on number of logical processors.</param>
            /// <param name="maximumTokenCountPerDocument">The threshold of maximum count of tokens per doc.</param>
            /// <param name="numberOfSummaryTermsPerTopic">The number of words to summarize the topic.</param>
            /// <param name="numberOfBurninIterations">The number of burn-in iterations.</param>
            /// <param name="resetRandomGenerator">Reset the random number generator for each document.</param>
            public ColumnOptions(string name,
                string inputColumnName = null,
                int numberOfTopics = LatentDirichletAllocationEstimator.Defaults.NumberOfTopics,
                float alphaSum = LatentDirichletAllocationEstimator.Defaults.AlphaSum,
                float beta = LatentDirichletAllocationEstimator.Defaults.Beta,
                int samplingStepCount = LatentDirichletAllocationEstimator.Defaults.SamplingStepCount,
                int maximumNumberOfIterations = LatentDirichletAllocationEstimator.Defaults.MaximumNumberOfIterations,
                int likelihoodInterval = LatentDirichletAllocationEstimator.Defaults.LikelihoodInterval,
                int numberOfThreads = LatentDirichletAllocationEstimator.Defaults.NumberOfThreads,
                int maximumTokenCountPerDocument = LatentDirichletAllocationEstimator.Defaults.MaximumTokenCountPerDocument,
                int numberOfSummaryTermsPerTopic = LatentDirichletAllocationEstimator.Defaults.NumberOfSummaryTermsPerTopic,
                int numberOfBurninIterations = LatentDirichletAllocationEstimator.Defaults.NumberOfBurninIterations,
                bool resetRandomGenerator = LatentDirichletAllocationEstimator.Defaults.ResetRandomGenerator)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValueOrNull(inputColumnName);
                Contracts.CheckParam(numberOfTopics > 0, nameof(numberOfTopics), "Must be positive.");
                Contracts.CheckParam(samplingStepCount > 0, nameof(samplingStepCount), "Must be positive.");
                Contracts.CheckParam(maximumNumberOfIterations > 0, nameof(maximumNumberOfIterations), "Must be positive.");
                Contracts.CheckParam(likelihoodInterval > 0, nameof(likelihoodInterval), "Must be positive.");
                Contracts.CheckParam(numberOfThreads >= 0, nameof(numberOfThreads), "Must be positive or zero.");
                Contracts.CheckParam(maximumTokenCountPerDocument > 0, nameof(maximumTokenCountPerDocument), "Must be positive.");
                Contracts.CheckParam(numberOfSummaryTermsPerTopic > 0, nameof(numberOfSummaryTermsPerTopic), "Must be positive");
                Contracts.CheckParam(numberOfBurninIterations >= 0, nameof(numberOfBurninIterations), "Must be non-negative.");

                Name = name;
                InputColumnName = inputColumnName ?? name;
                NumberOfTopics = numberOfTopics;
                AlphaSum = alphaSum;
                Beta = beta;
                SamplingStepCount = samplingStepCount;
                NumberOfIterations = maximumNumberOfIterations;
                LikelihoodInterval = likelihoodInterval;
                NumberOfThreads = numberOfThreads;
                MaximumTokenCountPerDocument = maximumTokenCountPerDocument;
                NumberOfSummaryTermsPerTopic = numberOfSummaryTermsPerTopic;
                NumberOfBurninIterations = numberOfBurninIterations;
                ResetRandomGenerator = resetRandomGenerator;
            }
        }

        /// <summary>
        /// Trains and returns a <see cref="LatentDirichletAllocationTransformer"/>.
        /// </summary>
        public LatentDirichletAllocationTransformer Fit(IDataView input)
        {
            return LatentDirichletAllocationTransformer.TrainLdaTransformer(_host, input, _columns.ToArray());
        }
    }
}
