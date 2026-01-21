namespace Catalyst
{
    /// <summary>
    /// Specifies the part-of-speech tags based on the Universal Dependencies project.
    /// </summary>
    public enum PartOfSpeech
    {
        /// <summary>No part-of-speech tag.</summary>
        NONE = 0,
        /// <summary>Adjective.</summary>
        ADJ, // adjective
        /// <summary>Adposition.</summary>
        ADP, // adposition
        /// <summary>Adverb.</summary>
        ADV, // adverb
        /// <summary>Auxiliary.</summary>
        AUX, // auxiliary
        /// <summary>Coordinating conjunction.</summary>
        CCONJ, // coordinating conjunction
        /// <summary>Determiner.</summary>
        DET, // determiner
        /// <summary>Interjection.</summary>
        INTJ, // interjection
        /// <summary>Noun.</summary>
        NOUN, // noun
        /// <summary>Numeral.</summary>
        NUM, // numeral
        /// <summary>Particle.</summary>
        PART, // particle
        /// <summary>Pronoun.</summary>
        PRON, // pronoun
        /// <summary>Proper noun.</summary>
        PROPN, // proper noun
        /// <summary>Punctuation.</summary>
        PUNCT, // punctuation
        /// <summary>Subordinating conjunction.</summary>
        SCONJ, // subordinating conjunction
        /// <summary>Symbol.</summary>
        SYM, // symbol
        /// <summary>Verb.</summary>
        VERB, // verb
        /// <summary>Other.</summary>
        X, // other
    }
}
