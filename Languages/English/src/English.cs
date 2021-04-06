using System;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class English
    {
        public static void Register()
        {
            ObjectStore.OverrideModel(new AveragePerceptronTagger(Language.English, 0).GetStoredObjectInfo(), async () => await ResourceLoader.LoadAsync(typeof(English).Assembly, "pos.binz", async (s) => { var a = new AveragePerceptronTagger(Language.English, 0, "");  await a.LoadAsync(s); return a; }));

            //Pipeline.Register
            
            //need a way to register on the Store the default models 
            //that will be loaded automatically by name and how to load them from a Lazy object instead
            
            //i.e. somehting like:
            
            //var lazyPOSModel = new Lazy<object>(() => LoadEnglishPOSModel())
            
            //Storage.RegisterDefaultModel(Language.English, "part-of-speech", lazyPOSModel);            
        }

        public sealed class Lemmatizer
        {

        }

        public sealed class StopWords
        {

        }

        
        public sealed class TokenizerExceptions 
        { 
        }
    }

}
