using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Mosaik.Core;

namespace Catalyst
{
    public class OnlineRepositoryStorage : IStorage
    {
        public IStorage Disk { get; }

        private static HttpClient Client;
        private Uri RepositoryAddress;

        private ConcurrentDictionary<string, (StoredObjectInfo objInfo, bool compressed)> MapPathToObjectInfo = new ConcurrentDictionary<string, (StoredObjectInfo objInfo, bool compressed)>();

        public OnlineRepositoryStorage(IStorage disk, string onlineRepository = "https://models.curiosity.ai/", HttpClient client = null)
        {
            if(!onlineRepository.EndsWith("/")) { onlineRepository = onlineRepository + "/"; }

            Disk = disk;
            Client = client ?? new HttpClient();
            RepositoryAddress = new Uri(onlineRepository);
        }

        public string BasePath => Disk.BasePath;

        public string TempPath => Disk.TempPath;

        public Task AppendAsync(Stream data, string path) => Disk.AppendAsync(data, path);

        public void CreateDirectory(string path) => Disk.CreateDirectory(path);

        public Task<bool> DeleteAsync(string path) => Disk.DeleteAsync(path);

        public void DeleteEmptyDirectories(string path) => Disk.DeleteEmptyDirectories(path);

        public (FileStream stream, string fileName) GetSharedTempStream(string path, long expectedLength = -1) => Disk.GetSharedTempStream(path, expectedLength);

        public FileStream GetTempStream() => Disk.GetTempStream();

        public IEnumerable<string> ListFiles(string path, string pattern, SearchOption searchOption) => Disk.ListFiles(path, pattern, searchOption);

        public Task<LockedMemoryMappedFile> OpenMemoryMappedFileAsync(string path) => Disk.OpenMemoryMappedFileAsync(path);

        public Task PutAsync(Stream data, string path) => Disk.PutAsync(data, path);

        public Task<Stream> OpenStreamAsync(string path, FileAccess access) => Disk.OpenStreamAsync(path, access);

        public Task<FileInfo> GetFileInfoAsync(string path) => Disk.GetFileInfoAsync(path);

        public async Task<LockedStream> OpenLockedStreamAsync(string path, FileAccess access)
        {
            if(access == FileAccess.Read && await Disk.ExistsAsync(path))
            {
                return await Disk.OpenLockedStreamAsync(path, access);
            }
            else if(access == FileAccess.Read)
            {
                //First try to download file
                var (objInfo, compressed) = MapPathToObjectInfo[path];
                if(await ExistsOnlineAsync(objInfo, compressed))
                {
                    using(var target = await Disk.OpenLockedStreamAsync(path, FileAccess.Write))
                    using (var f = await DownloadFileAsync(objInfo, compressed))
                    {
                        await f.CopyToAsync(target);
                        await target.FlushAsync();
                    }
                    return await Disk.OpenLockedStreamAsync(path, access);
                }
                else
                {
                    throw new FileNotFoundException(path);
                }
            }
            else
            {
                throw new FileNotFoundException(path);
            }
        }

        private async Task<Stream> DownloadFileAsync(StoredObjectInfo objInfo, bool compressed)
        {
            var resp = await Client.GetAsync(RepositoryAddress + $"/api/models?modelType={objInfo.ModelType}&language={Languages.EnumToCode(objInfo.Language)}&version={objInfo.Version}&tag={objInfo.Tag}&compress={compressed}");
            if (resp.IsSuccessStatusCode)
            {
                return await resp.Content.ReadAsStreamAsync();
            }
            else
            {
                throw new Exception("Invalid response from repository:" + resp.StatusCode);
            }
        }

        public async Task<bool> ExistsAsync(string path)
        {
            if(await Disk.ExistsAsync(path))
            {
                return true;
            }
            else
            {
                var (objInfo, compressed) = MapPathToObjectInfo[path];
                return await ExistsOnlineAsync(objInfo, compressed);
            }
        }

        private async Task<bool> ExistsOnlineAsync(StoredObjectInfo objInfo, bool compressed)
        {
            var resp = await Client.GetAsync(RepositoryAddress + $"/api/models/exists?modelType={objInfo.ModelType}&language={Languages.EnumToCode(objInfo.Language)}&version={objInfo.Version}&tag={objInfo.Tag}&compress={compressed}");
            if (resp.IsSuccessStatusCode)
            {
                return true;
            }
            else if(resp.StatusCode == System.Net.HttpStatusCode.NoContent)
            {
                return false;
            }
            else
            {
                throw new Exception("Invalid response from repository:" + resp.StatusCode);
            }
        }

        public string GetPath(IStorageTarget storeTarget, Language language, string modelType, int version, string tag, bool compressed)
        {
            var pathOnDisk = Disk.GetPath(storeTarget, language, modelType, version, tag, compressed);
            var objInfo = new StoredObjectInfo(modelType, language, version, tag);
            MapPathToObjectInfo[pathOnDisk] = (objInfo, compressed);
            return pathOnDisk;
        }
    }
}