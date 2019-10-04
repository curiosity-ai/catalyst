using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.WindowsAzure.Storage.Blob;

using Mosaik.Core;
using UID;

namespace Catalyst.Azure
{
    public class AzureBlobStorage : IStorage
    {
        public string BasePath => "";

        public string TempPath { get; private set; }

        private readonly AzureStorageContext _azureContext;
        private readonly string _containerName;
        private readonly string _contentType;

        private CloudBlobContainer _container;

        public AzureBlobStorage(AzureStorageContext azureContext, string containerName, string tempDirectory = null)
        {
            _azureContext = azureContext;
            _containerName = containerName;
            _contentType = "application/octet-stream";
            TempPath = tempDirectory ?? Path.GetTempPath();
            CreateDirectory(TempPath);
        }

        protected async Task<CloudBlobContainer> GetContainerAsync()
        {
            if (_container == null) {
                var blobClient = _azureContext.GetBlobClient();
                _container = blobClient.GetContainerReference(_containerName);
                await _container.CreateIfNotExistsAsync();
            }
            return _container;
        }

        public async Task AppendAsync(Stream data, string path)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);
            blockBlob.Properties.ContentType = _contentType;
            await blockBlob.UploadFromStreamAsync(data);
        }

        public void CreateDirectory(string path)
        {
            Directory.CreateDirectory(path);
        }

        public async Task<bool> DeleteAsync(string path)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);
            return await blockBlob.DeleteIfExistsAsync();
        }

        public void DeleteEmptyDirectories(string path)
        {
            // not used
        }

        public async Task<bool> ExistsAsync(string path)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);
            return await blockBlob.ExistsAsync();
        }

        public string GetDataPath(Language language, string modelType, int version, string file)
        {
            var path = Path.Combine("Data", Languages.EnumToCode(language), PathExtensions.GetValidPathName(modelType), $"v{version:000000}");
            if (!string.IsNullOrWhiteSpace(file)) {
                path = Path.Combine(path, file);
            }
            return path;
        }

        public Task<FileInfo> GetFileInfoAsync(string path)
        {
            //not used
            throw new NotImplementedException();
        }

        public string GetPath(IStorageTarget storeTarget, Language language, string modelType, int version, string tag, bool compressed)
        {
            string path = storeTarget.GetPath(BasePath, language, PathExtensions.GetValidPathName(modelType), version, tag);
            tag = PathExtensions.GetValidFileName(tag);
            return Path.Combine(path, (string.IsNullOrWhiteSpace(tag) ? $"model-v{version:000000}.bin" : $"model-{tag}-v{version:000000}.bin") + (compressed ? "z" : ""));
        }

        public (FileStream stream, string fileName) GetSharedTempStream(string path, long expectedLength = -1)
        {
            var fn = Path.Combine(TempPath, path.ToLowerInvariant().Hash128().ToString() + ".tmp");
            if (File.Exists(fn) && new FileInfo(fn).Length == expectedLength) {
                return (new FileStream(fn, FileMode.Open, FileAccess.Read, FileShare.Read, 4096, FileOptions.RandomAccess), fn);
            }
            else {
                return (new FileStream(fn, FileMode.Create, FileAccess.ReadWrite, FileShare.Read, 4096, FileOptions.RandomAccess), fn);
            }
        }

        public FileStream GetTempStream()
        {
            return new FileStream(Path.Combine(TempPath, Guid.NewGuid().ToString().Replace("-", "") + ".tmp"), 
                    FileMode.CreateNew, FileAccess.ReadWrite, FileShare.None, 4096, FileOptions.DeleteOnClose);
        }

        public IEnumerable<string> ListFiles(string path, string pattern, SearchOption searchOption)
        {
            return ListFilesAsync(path, pattern, searchOption).GetAwaiter().GetResult();
        }

        public async Task<IEnumerable<string>> ListFilesAsync(string path, string pattern, SearchOption searchOption)
        {
            var container = await GetContainerAsync();

            BlobContinuationToken continuationToken = null;

            //`null` will have the query return the entire contents of the blob container
            int? maxResultsPerQuery = null;

            var result = new List<string>();
            do {
                var response = await container.ListBlobsSegmentedAsync(string.Empty, true, BlobListingDetails.None, maxResultsPerQuery, continuationToken, null, null);
                continuationToken = response.ContinuationToken;
                foreach (var blob in response.Results) {
                    result.Add(BlobUritoPath((blob as CloudBlob).Name)); 
                }
            } 
            while (continuationToken != null);

            return result;
        }

        private string BlobUritoPath(string uri)
        {
            return uri.ToString();
        }

        public async Task<LockedStream> OpenLockedStreamAsync(string path, FileAccess access)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);

            var blobStream = access == FileAccess.Read
                    ? await blockBlob.OpenReadAsync()
                    : await blockBlob.OpenWriteAsync();
            
            var result = LockedStream.FromStream(path, blobStream, access, null);
            return result;
        }

        public Task<LockedMemoryMappedFile> OpenMemoryMappedFileAsync(string path)
        {
            //not used
            throw new NotImplementedException();
        }

        public async Task<Stream> OpenStreamAsync(string path, FileAccess access)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);

            return access == FileAccess.Read
                    ? await blockBlob.OpenReadAsync()
                    : await blockBlob.OpenWriteAsync();
        }

        public async Task PutAsync(Stream data, string path)
        {
            var container = await GetContainerAsync();
            var blockBlob = container.GetBlockBlobReference(path);
            blockBlob.Properties.ContentType = _contentType;
            await blockBlob.UploadFromStreamAsync(data);
        }
    }
}
