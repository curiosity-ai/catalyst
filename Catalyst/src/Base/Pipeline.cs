using Microsoft.Extensions.Logging;
using Mosaik.Core;
using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using System.IO.Compression;
using System.Reflection;

namespace Catalyst
{
    [FormerName("Mosaik.NLU", "PipelineModelData")]
    public class PipelineData : StorableObjectData
    {
        public List<StoredObjectInfo> Processes { get; set; }
    }

    public class Pipeline : StorableObject<Pipeline, PipelineData>, ICanUpdateModel
    {
        private List<IProcess> Processes { get; set; } = new List<IProcess>();
        private Dictionary<Language, Neuralyzer> Neuralyzers { get; set; } = new Dictionary<Language, Neuralyzer>();

        private ReaderWriterLockSlim RWLock = new ReaderWriterLockSlim();

        public Pipeline(Language language = Language.Any, int version = 0, string tag = "") : base(language, version, tag)
        {
        }

        public Pipeline(IList<IProcess> processes, Language language = Language.Any, int version = 0, string tag = "") : this(language, version, tag)
        {
            Processes = processes.ToList();
        }

        public new static async Task<Pipeline> FromStoreAsync(Language language, int version, string tag)
        {
            var pipeline = new Pipeline(language, version, tag);
            await pipeline.LoadDataAsync();
            var set = new HashSet<string>();

            foreach (var md in pipeline.Data.Processes.ToArray()) //Copy here as we'll modify the list bellow if model not found
            {
                if (await md.ExistsAsync())
                {
                    try
                    {
                        if (set.Add(md.ToStringWithoutVersion()))
                        {
                            var process = (IProcess)await md.FromStoreAsync();
                            pipeline.Add(process);
                        }
                    }
                    catch (FileNotFoundException)
                    {
                        Logger.LogError($"Model not found on disk, ignoring: {md.ToString()}");
                        pipeline.Data.Processes.Remove(md);
                    }
                }
                else
                {
                    pipeline.Data.Processes.Remove(md);
                }
            }

            if (!pipeline.Processes.Any(p => p is ITokenizer))
            {
                pipeline.AddToBegin(new FastTokenizer(language)); //Fix bug that removed all tokenizers from the pipelines due to missing ExistsAsync
            }

            return pipeline;
        }

        public void PackTo(Stream outputStream)
        {
            UpdateProcessData();
            using (var zip = new ZipArchive(outputStream, ZipArchiveMode.Create,leaveOpen: true))
            {
                var infoEntry = zip.CreateEntry("info.bin");
                using (var s = infoEntry.Open())
                {
                    MessagePack.LZ4MessagePackSerializer.Serialize(s, new StoredObjectInfo(typeof(Pipeline).FullName, Language, Version, Tag));
                }

                var pipeEntry = zip.CreateEntry("pipeline.bin");
                using(var s = pipeEntry.Open())
                {
                    MessagePack.LZ4MessagePackSerializer.Serialize(s, Data);
                }

                foreach (var process in Processes)
                {
                    var dataProperty = process.GetType().GetProperty(nameof(Spotter.Data)); //Uses spotter just to get the name of the .Data property from StorableObject<,> instead of hard-coding
                    if (dataProperty is object)
                    {
                        var data = dataProperty.GetValue(process, null) as StorableObjectData;
                        if (data is object)
                        {
                            var type = data.GetType();
                            var correctType = Convert.ChangeType(data, type);

                            var entry = zip.CreateEntry(new StoredObjectInfo(process).ToString() + ".bin");
                            using (var s = entry.Open())
                            {
                                MessagePack.LZ4MessagePackSerializer.Serialize(s, correctType);
                            }
                        }
                    }
                }
            }
        }

        public static async Task<Pipeline> LoadFromPackedAsync(Stream inputStream)
        {
            using (var zip = new ZipArchive(inputStream, ZipArchiveMode.Read, leaveOpen: true))
            {
                var infoEntry = zip.GetEntry("info.bin");
                StoredObjectInfo info;
                using (var s = infoEntry.Open())
                {
                    info = MessagePack.LZ4MessagePackSerializer.Deserialize<StoredObjectInfo>(s);
                }

                var pipeline = new Pipeline(info.Language, info.Version, info.Tag);

                var pipeEntry = zip.GetEntry("pipeline.bin");
                using (var s = pipeEntry.Open())
                {
                    pipeline.Data = MessagePack.LZ4MessagePackSerializer.Deserialize<PipelineData>(s);
                }

                foreach(var process in pipeline.Data.Processes)
                {
                    if (ObjectStore.TryGetType(process.ModelType, out var type))
                    {
                        object model = null;

                        foreach (var constructorInfo in type.GetConstructors().Concat(type.GetConstructors(BindingFlags.Instance | BindingFlags.NonPublic)))
                        {
                            var parameters = constructorInfo.GetParameters();
                            if (parameters.Length == 3
                                && parameters[0].ParameterType == typeof(Language)
                                && parameters[1].ParameterType == typeof(int)
                                && parameters[2].ParameterType == typeof(string))
                            {
                                //Found a good one
                                model = constructorInfo.Invoke(new object[] { process.Language, process.Version, process.Tag });
                                break;
                            }

                            if (parameters.Length == 2
                                && parameters[0].ParameterType == typeof(Language)
                                && parameters[1].ParameterType == typeof(int))
                            {
                                //Found a good one
                                model = constructorInfo.Invoke(new object[] { process.Language, process.Version });
                                break;
                            }
                            if (parameters.Length == 1
                                && parameters[0].ParameterType == typeof(Language))
                            {
                                //Found a good one
                                model = constructorInfo.Invoke(new object[] { process.Language });
                                break;
                            }
                        }

                        var dataProperty = model.GetType().GetProperty(nameof(Spotter.Data)); //Uses spotter just to get the name of the .Data property from StorableObject<,> instead of hard-coding
                        if(dataProperty is object)
                        {
                            var entry = zip.GetEntry(process.ToString() + ".bin");
                            using (var s = entry.Open())
                            using (var ms = new MemoryStream())
                            {
                                await s.CopyToAsync(ms);
                                var bytes = ms.ToArray();
                                var deserializer = SerializationHelper.CreateDeserializer(dataProperty.PropertyType);
                                var modelData = deserializer.Invoke(bytes);

                                dataProperty.GetSetMethod().Invoke(model, new[] { modelData });
                            }
                        }

                        if (model is object)
                        {
                            pipeline.Add((IProcess)model);
                        }
                    }
                }

                return pipeline;
            }
        }

        public static async Task<bool> CheckIfHasModelAsync(Language language, int version, string tag, StoredObjectInfo model, bool matchVersion = true)
        {
            var a = new Pipeline(language, version, tag);
            await a.LoadDataAsync();
            return a.Data.Processes.Any(p => p.Language == model.Language && p.ModelType == model.ModelType && p.Tag == model.Tag && (!matchVersion || p.Version == model.Version));
        }

        public override async Task StoreAsync()
        {
            UpdateProcessData();
            await base.StoreAsync();
        }

        private void UpdateProcessData()
        {
            Data.Processes = new List<StoredObjectInfo>();
            var set = new HashSet<string>();
            foreach (var p in Processes)
            {
                var md = new StoredObjectInfo(p.Type, p.Language, p.Version, p.Tag);
                if (set.Add(md.ToStringWithoutVersion()))
                {
                    Data.Processes.Add(md);
                }
            }
        }

        public void AddSpecialCase(string word, TokenizationException exception)
        {
            foreach (var p in Processes)
            {
                if (p is FastTokenizer st)
                {
                    st.AddSpecialCase(word, exception);
                }
            }
        }

        public Pipeline UseNeuralyzer(Neuralyzer neuralyzer)
        {
            RWLock.EnterWriteLock();
            try
            {
                Neuralyzers[neuralyzer.Language] = neuralyzer;
                return this;
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        public Pipeline UseNeuralyzers(IEnumerable<Neuralyzer> neuralyzeres)
        {
            RWLock.EnterWriteLock();
            try
            {
                Neuralyzers.Clear();
                foreach (var n in neuralyzeres)
                {
                    Neuralyzers[n.Language] = n;
                }
                return this;
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        public void RemoveAllNeuralizers()
        {
            RWLock.EnterWriteLock();
            try
            {
                Neuralyzers.Clear();
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        private void TryImportSpecialCases(IProcess process)
        {
            if (process is IHasSpecialCases)
            {
                foreach (FastTokenizer st in Processes.Where(p => p is FastTokenizer))
                {
                    if (process.Language == Language.Any || st.Language == process.Language)
                    {
                        st.ImportSpecialCases(process);
                    }
                }
            }
        }

        public Pipeline AddToBegin(IProcess process)
        {
            RWLock.EnterWriteLock();
            try
            {
                var tmp = Processes.ToList();
                Processes.Clear();
                Processes.Add(process);
                Processes.AddRange(tmp);
                TryImportSpecialCases(process);
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
            return this;
        }

        public void RemoveAll(Predicate<IProcess> p)
        {
            RWLock.EnterWriteLock();

            try
            {
                Processes.RemoveAll(p);
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        public async Task<Pipeline> AddAsync(Task<IProcess> modelLoadTask) => Add(await modelLoadTask);

        public Pipeline Add(IProcess process)
        {
            RWLock.EnterWriteLock();
            try
            {
                //Ensure correct order when adding new processes

                var normalizers       = Processes.Where(p => p is INormalizer).ToList();
                var tokenizers        = Processes.Where(p => p is ITokenizer).ToList();
                var sentenceDetectors = Processes.Where(p => p is ISentenceDetector).ToList();
                var others            = Processes.Except(normalizers).Except(tokenizers).Except(sentenceDetectors).ToList();

                if (process is INormalizer)
                {
                    normalizers.Add(process);
                }
                else if (process is ITokenizer)
                {
                    tokenizers.Add(process);
                }
                else if (process is ISentenceDetector)
                {
                    sentenceDetectors.Add(process);
                }
                else
                {
                    others.Add(process);
                }

                Processes = normalizers.Concat(tokenizers).Concat(sentenceDetectors).Concat(others).ToList();

                TryImportSpecialCases(process);
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
            return this;
        }

        public void ReplaceWith(Pipeline newPipeline)
        {
            RWLock.EnterWriteLock();
            try
            {
                Data.Processes = newPipeline.Data.Processes;
                Processes = newPipeline.Processes.ToList();
                Version = newPipeline.Version;
            }
            finally
            {
                RWLock.ExitWriteLock();
            }
        }

        public static Pipeline For(Language language, bool sentenceDetector = true, bool tagger = true)
        {
            return ForAsync(language, sentenceDetector, tagger).WaitResult();
        }

        public static async Task<Pipeline> ForAsync(Language language, bool sentenceDetector = true, bool tagger = true)
        {
            var p = await TokenizerForAsync(language);
            if (tagger) { p.Add(await AveragePerceptronTagger.FromStoreAsync(language, 0, "")); }
            return p;
        }

        public static async Task<Pipeline> ForManyAsync(IEnumerable<Language> languages, bool sentenceDetector = true, bool tagger = true)
        {
            var processes = new List<IProcess>();
            foreach (var language in languages)
            {
                var tmp = await TokenizerForAsync(language);
                processes.AddRange(tmp.Processes);
                if (tagger) { processes.Add(await AveragePerceptronTagger.FromStoreAsync(language, -1, "")); }
            }
            var p = new Pipeline(processes) { Language = Language.Any };
            return p;
        }

        public static Pipeline TokenizerFor(Language language)
        {
            return TokenizerForAsync(language).WaitResult();
        }

        public static async Task<Pipeline> TokenizerForAsync(Language language)
        {
            var p = new Pipeline() { Language = language };
            p.Add(new FastTokenizer(language));

            IProcess sd = null;

            try
            {
                //Uses english sentence detector as a default
                sd = await SentenceDetector.FromStoreAsync((language == Language.Any) ? Language.English : language, -1, "");
                p.Add(sd);
            }
            catch
            {
                Logger.LogWarning("Could not find sentence detector model for language {LANGUAGE}. Falling back to english model", language);
            }

            if (sd is null)
            {
                try
                {
                    sd = await SentenceDetector.FromStoreAsync(Language.English, -1, "");
                    p.Add(sd);
                }
                catch
                {
                    Logger.LogWarning("Could not find sentence detector model for language {LANGUAGE}. Continuing without one", Language.English);
                }
            }

            return p;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public IDocument ProcessSingle(IDocument document)
        {
            RWLock.EnterReadLock();
            try
            {
                return ProcessSingleWithoutLocking(document);
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private IDocument ProcessSingleWithoutLocking(IDocument document)
        {
            if (document.Length > 0)
            {
                foreach (var p in Processes)
                {
                    if (p.Language != Language.Any && document.Language != Language.Any && p.Language != document.Language) { continue; }
                    p.Process(document);
                }

                //Apply any neuralizer registered for any language, or the document language
                if (Neuralyzers.TryGetValue(Language.Any, out var neuralyzerAny)) { neuralyzerAny.Process(document); }
                if (Neuralyzers.TryGetValue(document.Language, out var neuralyzerLang)) { neuralyzerLang.Process(document); }
            }
            return document;
        }

        public IEnumerable<IDocument> ProcessSingleThread(IEnumerable<IDocument> documents)
        {
            RWLock.EnterReadLock();

            var sw = Stopwatch.StartNew();
            long docsCount = 0, spansCount = 0, tokensCount = 0;

            Logger.LogTrace("Started pipeline single thread processing");

            var buffer = new List<IDocument>();

            foreach (var block in documents.Split(1000))
            {
                RWLock.EnterReadLock(); //Acquire the read lock only for the duration of the block processing, not during the yield return
                try
                {
                    foreach (var doc in block)
                    {
                        IDocument d;
                        try
                        {
                            d = ProcessSingleWithoutLocking(doc);

                            Interlocked.Add(ref spansCount, doc.SpansCount);
                            Interlocked.Add(ref tokensCount, doc.TokensCount);

                            if (Interlocked.Increment(ref docsCount) % 10_000 == 0)
                            {
                                var elapsed = sw.Elapsed.TotalSeconds;
                                var kts = tokensCount / elapsed / 1000;
                                Logger.LogInformation("Parsed {DOCS} documents, {SPANS} sentences and {TOKENS} tokens in {ELAPSED:0.00} seconds at {KTS} kTokens/second", docsCount, spansCount, tokensCount, (int)elapsed, (int)kts);
                            }
                        }
                        catch (Exception E)
                        {
                            Logger.LogError(E, "Error parsing document");
                            d = null;
                        }

                        if (d is object)
                        {
                            buffer.Add(d);
                        }
                    }
                }
                finally
                {
                    RWLock.ExitReadLock(); //Return the lock here, otherwise we don't have control due to the yield return pattern
                }

                foreach(var d in block)
                {
                    yield return d;
                }
            }
        }

        public IEnumerable<IDocument> Process(IEnumerable<IDocument> documents, ParallelOptions parallelOptions = default)
        {
            var sw = Stopwatch.StartNew();
            long docsCount = 0, spansCount = 0, tokensCount = 0;
            double kts;

            var enumerator = documents.GetEnumerator();
            var buffer = new List<IDocument>();

            parallelOptions = parallelOptions ?? new ParallelOptions();

            using (var m = new Measure(Logger, "Parsing documents", logOnlyDuration:true))
            {
                while (enumerator.MoveNext())
                {
                    buffer.Add(enumerator.Current);

                    if (buffer.Count >= 10_000)
                    {

                        RWLock.EnterReadLock(); //Acquire the read lock only for the duration of the processing, not during the yield return
                        try
                        {
                            Parallel.ForEach(buffer, parallelOptions, (doc) => ProcessSingleWithoutLocking(doc));
                        }
                        finally
                        {
                            RWLock.ExitReadLock(); //Return the lock here, otherwise we don't have control due to the yield return pattern
                        }

                        foreach (var doc in buffer) { spansCount += doc.SpansCount; tokensCount += doc.TokensCount; docsCount++; yield return doc; }
                        buffer.Clear();
                        kts = (double)tokensCount / m.ElapsedSeconds / 1000;
                        m.SetOperations(docsCount).EmitPartial($"{kts:n0} kTokens/second, found {spansCount:n0} sentences and {tokensCount:n0} tokens");
                    }
                }

                RWLock.EnterReadLock(); //Acquire the read lock only for the duration of the processing, not during the yield return
                try
                {
                    //Process any remaining
                    Parallel.ForEach(buffer, parallelOptions, (doc) => ProcessSingleWithoutLocking(doc));
                }
                finally
                {
                    RWLock.ExitReadLock(); //Return the lock here, otherwise we don't have control due to the yield return pattern
                }

                foreach (var doc in buffer) { spansCount += doc.SpansCount; tokensCount += doc.TokensCount; docsCount++; yield return doc; }
                buffer.Clear();
                kts = (double)tokensCount / m.ElapsedSeconds / 1000;
                m.SetOperations(docsCount).EmitPartial($"{kts:n0} kTokens/second, found {spansCount:n0} sentences and {tokensCount:n0} tokens");
            }
        }

        public IList<IModel> GetModelsList()
        {
            RWLock.EnterReadLock();
            try
            {
                return Processes.Select(p => (IModel)p).ToList();
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public List<StoredObjectInfo> GetModelsDescriptions()
        {
            RWLock.EnterReadLock();
            try
            {
                return Processes.Select(p => new StoredObjectInfo(p.Type, p.Language, p.Version, p.Tag)).ToList();
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public IEnumerable<(StoredObjectInfo Model, string[] EntityTypes)> GetPossibleEntityTypes()
        {
            RWLock.EnterReadLock();
            try
            {
                foreach (var p in Processes)
                {
                    if (p is IEntityRecognizer ire)
                    {
                        var md = new StoredObjectInfo(ire);
                        yield return (md, ire.Produces());
                    }
                }
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public bool HasModel(StoredObjectInfo model, bool matchVersion = true)
        {
            RWLock.EnterReadLock();

            try
            {
                return Processes.Any(p => p.Language == model.Language && p.Type == model.ModelType && p.
                Tag == model.Tag && (!matchVersion || p.Version == model.Version));
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public bool RemoveModel(StoredObjectInfo model)
        {
            //Removes any model matching the description, independent of the version
            RWLock.EnterReadLock();

            try
            {
                return Processes.RemoveAll(p => p.Language == model.Language && p.Type == model.ModelType && p.Tag == model.Tag) > 0;
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public bool HasModelToUpdate(StoredObjectInfo newModel)
        {
            RWLock.EnterReadLock();

            try
            {
                return Processes.Any(p => p.Language == newModel.Language && p.Type == newModel.ModelType && p.Tag == newModel.Tag && p.Version < newModel.Version);
            }
            finally
            {
                RWLock.ExitReadLock();
            }
        }

        public void UpdateModel(StoredObjectInfo newModel, object modelToBeUpdated)
        {
            if (modelToBeUpdated is IProcess process)
            {
                RWLock.EnterWriteLock();

                try
                {
                    var toBeReplaced = Processes.FirstOrDefault(p => p.Language == newModel.Language && p.Type == newModel.ModelType && p.Tag == newModel.Tag && p.Version < newModel.Version);

                    if (toBeReplaced is object)
                    {
                        var index = Processes.IndexOf(toBeReplaced);
                        if (index >= 0)
                        {
                            Processes[index] = process;
                        }
                    }
                }
                finally
                {
                    RWLock.ExitWriteLock();
                }
            }
            else
            {
                throw new InvalidOperationException("Invalid model to update");
            }
        }
    }
}