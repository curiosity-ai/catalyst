using Catalyst;
using Mosaik.Core;
using System;
using System.Threading.Tasks;

namespace Catalyst_Samples
{
    class Program
    {
        static void Main(string[] args)
        {
            try
            {
                SimpleTest().Wait();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"There was an exception: {ex.ToString()}");
            }
        }

        private static async Task SimpleTest()
        {
            Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));
            var nlp = await Pipeline.ForAsync(Language.English);
            var doc = new Document("The quick brown fox jumps over the lazy dog", Language.English);
            
           

            Console.WriteLine(doc.ToJson());
        }
    }
}
