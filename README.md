<img src="https://raw.githubusercontent.com/curiosity-ai/catalyst/master/Catalyst/catalyst.png?token=ACDCOAYAIML2KGJTHTJP27C5KGCEC"/>

<a href="https://curiosity.ai"><img src="https://curiosity.ai/assets/images/logos/curiosity.png" width="100" height="100" align="right" /></a>

_**catalyst**_ is a C# Natural Language Processing library built for speed. Inspired by [spaCy's design](https://spacy.io/), it brings pre-trained models, out-of-the box support for training word and document embeddings, and flexible entity recognition models.

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


## ✨ Getting Started

Using _**catalyst**_ is as simple as installing its [NuGet Package](https://www.nuget.org/packages/Catalyst), and setting the storage to use our online repository. This way, models will be lazy loaded either from disk or downloaded from our online repository.

```csharp
Storage.Current = new OnlineRepositoryStorage(new DiskStorage("catalyst-models"));
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



## 📖 Documentation (coming soon)

| Documentation     |                                                       |
| ----------------- | ----------------------------------------------------- |
| [Getting Started] | How to use _**catalyst**_ and its features.           |
| [API Reference]   | The detailed reference for _**catalyst**_'s API.      |
| [Contribute]      | How to contribute to _**catalyst**_ codebase.         |

[Getting Started]: https://catalyst.curiosity.ai/getting-started
[api reference]: https://catalyst.curiosity.ai/api
[contribute]: https://github.com/curiosity-ai/catalyst/blob/master/CONTRIBUTING.md
