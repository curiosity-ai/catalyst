using System;
using System.IO;
using System.Threading.Tasks;

using Xunit;
using FluentAssertions;

namespace Catalyst.Azure.Tests
{
    public class AzureBlobStorageTests
    {
        [Fact]
        public async Task Create_Write_Read()
        {
            var azureContext = new AzureStorageContext("UseDevelopmentStorage=true;");
            var storage = new AzureBlobStorage(azureContext, "test-catalyst-data");

            var buffer = new byte[40000];
            for (int i = 0; i < buffer.Length; i++) {
                buffer[i] = (byte)((i % 255) + 1);
            }
            var inStream = new MemoryStream(buffer);
            inStream.Position = 0;
            await storage.AppendAsync(inStream, "models/pos-taggers/en/01/model1");
            inStream.Position = 0;
            await storage.AppendAsync(inStream, "models/pos-taggers/uk/05/model2");
            inStream.Position = 0;
            await storage.AppendAsync(inStream, "models/ner/en/02/model3");
            inStream.Position = 0;
            await storage.AppendAsync(inStream, "models/ner/uk/07/model4");
            inStream.Position = 0;
            await storage.AppendAsync(inStream, "models/unidep/en/03/model5");

            var list = storage.ListFiles("", "", System.IO.SearchOption.AllDirectories);

            list.Should().NotBeEmpty().And.HaveCount(5);

            var outStream = await storage.OpenStreamAsync("models/unidep/en/03/model5", FileAccess.Read);
            var memoryStream = new MemoryStream(40000);
            outStream.CopyTo(memoryStream);
            var bytes = memoryStream.GetBuffer();

            bytes.Should().HaveCount(40000);

            bytes[256].Should().Be(2);
        }
    }
}
