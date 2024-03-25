[![Nuget](https://img.shields.io/nuget/v/Catalyst.svg?maxAge=0&colorB=brightgreen)](https://www.nuget.org/packages/Catalyst/) [![Build Status](https://dev.azure.com/curiosity-ai/mosaik/_apis/build/status/catalyst?branchName=master)](https://dev.azure.com/curiosity-ai/mosaik/_build/latest?definitionId=10&branchName=master)
[![BuiltWithDot.Net shield](https://builtwithdot.net/project/427/catalyst-nlp-csharp-spacy-embeddings/badge)](https://builtwithdot.net/project/427/catalyst-nlp-csharp-spacy-embeddings)

<img src="https://raw.githubusercontent.com/curiosity-ai/catalyst/master/Catalyst/catalyst.png?token=ACDCOAYAIML2KGJTHTJP27C5KGCEC"/>

<a href="https://curiosity.ai"><img src="https://curiosity.ai/media/cat.color.square.svg" width="100" height="100" align="right" /></a>

_**catalyst**_ is a C# Natural Language Processing library built for speed. Inspired by [spaCy's design](https://spacy.io/), it brings pre-trained models, out-of-the box support for training word and document embeddings, and flexible entity recognition models.

[![Gitter](https://badges.gitter.im/curiosityai/catalyst.svg)](https://gitter.im/curiosityai/catalyst?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## ⚡ Features
- Fast, modern pure-C# NLP library, supporting [.NET standard 2.0](https://docs.microsoft.com/en-us/dotnet/standard/net-standard)
- Cross-platform, runs anywhere [.NET core](https://dotnet.microsoft.com/download) is supported - Windows, Linux, macOS and even ARM
- Non-destructive [tokenization](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/Base/FastTokenizer.cs), >99.9% [RegEx-free](https://blog.codinghorror.com/regex-performance/), >1M tokens/s on a modern CPU
- Named Entity Recognition ([gazeteer](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/EntityRecognition/Spotter.cs), [rule-based](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/EntityRecognition/PatternSpotter.cs) & [perceptron-based](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/EntityRecognition/AveragePerceptronEntityRecognizer.cs))
- Pre-trained models based on [Universal Dependencies](https://universaldependencies.org/) project
- Custom models for learning [Abbreviations](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/Special/AbbreviationCapturer.cs) & [Senses](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/EntityRecognition/Spotter.cs#L214)
- Out-of-the-box support for training [FastText](https://fasttext.cc/) and [StarSpace](https://github.com/facebookresearch/StarSpace) embeddings (pre-trained models coming soon)
- Part-of-speech tagging
- Language detection using [FastText](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/Special/FastTextLanguageDetector.cs) or [cld3](https://github.com/curiosity-ai/catalyst/blob/master/Catalyst/src/Models/Special/LanguageDetector.cs)
- Efficient binary serialization based on [MessagePack](https://github.com/neuecc/MessagePack-CSharp/)
- Pre-built models for [language packages](https://www.nuget.org/packages?q=catalyst.models) ✨
- Lemmatization ✨ (using lookup tables ported from [spaCy](https://github.com/explosion/spacy-lookups-data))


## New: Language Packages ✨
We're migrating our model repository to use NuGet packages for all language-specific data and models. 

You can find all  new language packages [here](https://www.nuget.org/packages?q=catalyst.models). 

The new models are trained on the latest release of [Universal Dependencies v2.7](https://universaldependencies.org/).

This is technically not a breaking change *yet*, but our online repository will be deprecated in the near future - so you should migrate to the new NuGet packages.

When using the new model packages, you can usually remove this line from your code: `Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));`, or replace it with `Storage.Current = new DiskStorage("catalyst-models")` if you are storing your own models locally.

We've also added the option to store and load models using streams:
`````csharp
// Creates and stores the model
var isApattern = new PatternSpotter(Language.English, 0, tag: "is-a-pattern", captureTag: "IsA");
isApattern.NewPattern(
    "Is+Noun",
    mp => mp.Add(
        new PatternUnit(P.Single().WithToken("is").WithPOS(PartOfSpeech.VERB)),
        new PatternUnit(P.Multiple().WithPOS(PartOfSpeech.NOUN, PartOfSpeech.PROPN, PartOfSpeech.AUX, PartOfSpeech.DET, PartOfSpeech.ADJ))
));
using(var f = File.OpenWrite("my-pattern-spotter.bin"))
{
    await isApattern.StoreAsync(f);
}

// Load the model back from disk
var isApattern2 = new PatternSpotter(Language.English, 0, tag: "is-a-pattern", captureTag: "IsA");

using(var f = File.OpenRead("my-pattern-spotter.bin"))
{
    await isApattern2.LoadAsync(f);
}
`````


## ✨ Getting Started

Using _**catalyst**_ is as simple as installing its [NuGet Package](https://www.nuget.org/packages/Catalyst), and setting the storage to use our online repository. This way, models will be lazy loaded either from disk or downloaded from our online repository. **Check out also some of the [sample projects](https://github.com/curiosity-ai/catalyst/tree/master/samples)** for more examples on how to use _**catalyst**_.


```csharp
Catalyst.Models.English.Register(); //You need to pre-register each language (and install the respective NuGet Packages)

Storage.Current = new DiskStorage("catalyst-models");
var nlp = await Pipeline.ForAsync(Language.English);
var doc = new Document("The quick brown fox jumps over the lazy dog", Language.English);
nlp.ProcessSingle(doc);
Console.WriteLine(doc.ToJson());
```

You can also take advantage of C# lazy evaluation and native multi-threading support to process a large number of documents in parallel:

```csharp
var docs = GetDocuments();
var parsed = nlp.Process(docs);
DoSomething(parsed);

IEnumerable<IDocument> GetDocuments()
{
    //Generates a few documents, to demonstrate multi-threading & lazy evaluation
    for(int i = 0; i < 1000; i++)
    {
        yield return new Document("The quick brown fox jumps over the lazy dog", Language.English);
    }
}

void DoSomething(IEnumerable<IDocument> docs)
{
    foreach(var doc in docs)
    {
        Console.WriteLine(doc.ToJson());
    }
}
```

Training a new [FastText](https://fasttext.cc/) [word2vec](https://en.wikipedia.org/wiki/Word2vec) embedding model is as simple as this:

```csharp
var nlp = await Pipeline.ForAsync(Language.English);
var ft = new FastText(Language.English, 0, "wiki-word2vec");
ft.Data.Type = FastText.ModelType.CBow;
ft.Data.Loss = FastText.LossType.NegativeSampling;
ft.Train(nlp.Process(GetDocs()));
ft.StoreAsync();
```

For fast embedding search, we have also released a C# version of the ["Hierarchical Navigable Small World" (HNSW)](https://arxiv.org/abs/1603.09320) algorithm on [NuGet](https://www.nuget.org/packages/HNSW/), based on our fork of Microsoft's [HNSW.Net](https://github.com/curiosity-ai/hnsw.net). We have also released a C# version of the "Uniform Manifold Approximation and Projection" ([UMAP](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html)) algorithm for dimensionality reduction on [GitHub](https://github.com/curiosity-ai/umap-csharp) and on [NuGet](https://www.nuget.org/packages/UMAP/).



## 📖 Links

| Documentation     |                                                           |
| ----------------- | --------------------------------------------------------- |
| [Contribute]      | How to contribute to _**catalyst**_ codebase.             |
| [Samples]         | Sample projects demonstrating _**catalyst**_ capabilities |
| [![Gitter](https://badges.gitter.im/curiosityai/catalyst.svg)](https://gitter.im/curiosityai/catalyst?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)  | Join our gitter channel                                    |

[Contribute]: https://github.com/curiosity-ai/catalyst/blob/master/CONTRIBUTING.md
[Samples]: https://github.com/curiosity-ai/catalyst/tree/master/samples
