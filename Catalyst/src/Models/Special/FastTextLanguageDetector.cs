//using MessagePack;

using Mosaik.Core;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    [FormerName("Mosaik.NLU.Models", "VectorizerLanguageDetectorModel")]
    public class FastTextLanguageDetectorData : StorableObjectData
    {
    }

    [FormerName("Mosaik.NLU.Models", "VectorizerLanguageDetector")]
    public class FastTextLanguageDetector : StorableObject<FastTextLanguageDetector, FastTextLanguageDetectorData>, IProcess
    {
        private SpaceTokenizer Tokenizer;
        private FastText Model;
        private NumberToWordNormalizer NumberNormalizer = new NumberToWordNormalizer() { ReplacementText = "" };

        public FastTextLanguageDetector(int version) : base(Language.Any, version, nameof(FastTextLanguageDetector), compress: false)
        {
            Model = new FastText(Language.Any, version, "language-detector");
            Model.Data.Type = FastText.ModelType.Supervised;
            Model.Data.MaximumWordNgrams = 0;
            Model.Data.MinimumNgrams = 2;
            Model.Data.MaximumNgrams = 5;
            Model.Data.VectorQuantization = QuantizationType.None;
            Model.Data.LearningRate = 0.1f;
            Model.Data.Epoch = 50;
            Model.Data.Dimensions = 16;
            Model.Data.IgnoreCase = false;
            Model.Data.Loss = FastText.LossType.NegativeSampling;
            Model.Data.MinimumCount = 5;

            Tokenizer = new SpaceTokenizer();
        }

        public static Task<FastTextLanguageDetector> LoadAsync(int version)
        {
            return FromStoreAsync(Language.Any, version, "");
        }

        public new static async Task<FastTextLanguageDetector> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new FastTextLanguageDetector(version);

            //Because we use the model name as the tag of this model, we've to check for formernames here
            try
            {
                await a.LoadDataAsync();
            }
            catch (FileNotFoundException)
            {
                if (ObjectStore.TryGetFormerNames(nameof(FastTextLanguageDetector), out var formerNames))
                {
                    var correctTag = a.Tag;
                    foreach (var formerName in formerNames)
                    {
                        try
                        {
                            a.Tag = formerName;
                            await a.LoadDataAsync();
                            a.Tag = correctTag;
                            break;
                        }
                        catch (FileNotFoundException)
                        {
                            //ignore
                        }
                    }
                }
            }

            a.Model = await FastText.FromStoreAsync_Internal(Language.Any, version, "language-detector", bufferedMatrix: false);

            return a;
        }

        public new static async Task<bool> DeleteAsync(Language language, int version, string tag)
        {
            var a = new FastTextLanguageDetector(version);
            bool deleted = false;
            deleted |= await FastText.DeleteAsync(Language.Any, version, "language-detector");
            deleted |= await a.DeleteDataAsync();
            return deleted;
        }

        public void Process(IDocument document)
        {
            if (document.Length == 0 || (document.Language != Language.Unknown && document.Language != Language.Any)) { return; } //Don't try to identify documents that already have their language set or is empty

            IDocument tempDocument = document;

            if (document.SpansCount == 0) // Have to tokenize temporarily the document
            {
                if (document.Length > 1000)
                {
                    tempDocument = new Document(document.Value.Substring(0, 1000));
                }
                else
                {
                    tempDocument = new Document(document.Value);
                }
                Tokenizer.Process(tempDocument);
                NumberNormalizer.Process(tempDocument);
            }

            try
            {
                var tag = Model.PredictMax(tempDocument, 200);
                if (tag.label is null)
                {
                    document.Language = Language.Unknown;
                }
                else
                {
                    document.Language = Languages.CodeToEnum(tag.label);
                }
            }
            catch
            {
                document.Language = Language.Unknown;
            }
        }

        public void Train(IEnumerable<IDocument> documents)
        {
            Model.Train(documents);
        }

        public override async Task StoreAsync()
        {
            await Model.StoreAsync();
            await base.StoreAsync();
        }
    }
}