using Mosaik.Core;
using System;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace Catalyst.Tests
{
    public class PipelineTests
    {
        [Fact]
        public async Task Pack_Unpack()
        {
            Storage.Current = new Catalyst.OnlineRepositoryStorage(new DiskStorage("catalyst-models"));

            ObjectStore.OtherAssemblies.Add(typeof(Pipeline).Assembly);
            var pipeline1 = await Pipeline.ForAsync(Language.English);

            pipeline1.Version = 123;
            pipeline1.Tag = "Test";

            using(var ms = new MemoryStream())
            {
                pipeline1.PackTo(ms);    
                ms.Seek(0, SeekOrigin.Begin);
                var pipeline2 = await Pipeline.LoadFromPackedAsync(ms);

                Assert.Equal(pipeline1.Version, pipeline2.Version);
                Assert.Equal(pipeline1.Tag, pipeline2.Tag);
                Assert.Equal(string.Join(";", pipeline1.GetModelsDescriptions().Select(md => md.ToString())),
                             string.Join(";", pipeline2.GetModelsDescriptions().Select(md => md.ToString())));
            }
        }
    }
}
