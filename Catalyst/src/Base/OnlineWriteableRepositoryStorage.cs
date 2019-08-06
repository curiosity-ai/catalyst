using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Mosaik.Core;

namespace Catalyst
{
    public class OnlineWriteableRepositoryStorage : OnlineRepositoryStorage
    {
        public OnlineWriteableRepositoryStorage(IStorage disk, string token, string onlineRepository = "https://models.curiosity.ai/", HttpClient client = null) : base(disk, onlineRepository, client)
        {
            Client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        public override async Task PutAsync(Stream data, string path)
        {
            await Disk.PutAsync(data, path);
            data.Seek(0, SeekOrigin.Begin);
            var (objInfo, compressed) = MapPathToObjectInfo[path];
            await UploadFileAsync(path, objInfo, compressed, data);
        }

        public override async Task<LockedStream> OpenLockedStreamAsync(string path, FileAccess access)
        {
            if (access == FileAccess.Read && await ExistsAsync(path))
            {
                var stream = await Disk.OpenLockedStreamAsync(path, access);
                var (objInfo, compressed) = MapPathToObjectInfo[path];
                await UploadFileAsync(path, objInfo, compressed, stream);
                stream.Seek(0, SeekOrigin.Begin);
                return stream;
            }
            else
            {
                return await Disk.OpenLockedStreamAsync(path, access);
            }
        }

        private async Task UploadFileAsync(string path, ObjectInfo objInfo, bool compressed, Stream stream)
        {
            var fileContent = new StreamContent(stream);
            var form = new MultipartFormDataContent();
            fileContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
            form.Add(fileContent, "file", Path.GetFileName(path));

            var resp = await Client.PostAsync(RepositoryAddress + $"api/models?modelType={objInfo.ModelType}&language={Languages.EnumToCode(objInfo.Language)}&version={objInfo.Version}&tag={objInfo.Tag ?? ""}&compress={compressed}", form);

            if (!resp.IsSuccessStatusCode)
            {
                throw new Exception("Invalid response from repository:" + resp.StatusCode);
            }
        }
    }
}