// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst
{
    public static class PartOfSpeechHelpers
    {
        //Place the most common POS first
        public static PartOfSpeech[] All = new[] { PartOfSpeech.NOUN, PartOfSpeech.PROPN, PartOfSpeech.VERB }.Concat(Enum.GetValues(typeof(PartOfSpeech)).Cast<PartOfSpeech>()).Distinct().ToArray();

        public static readonly IDictionary<string, PartOfSpeech> EnglishPennToUniversal = new Dictionary<string, PartOfSpeech>()
        {
            ["NFP"] = PartOfSpeech.PUNCT, //"..." on Ontonotes corpus
            ["ADD"] = PartOfSpeech.X, //URLs on the Ontonotes corpus
            ["XX"] = PartOfSpeech.X,
            ["#"] = PartOfSpeech.SYM,
            ["$"] = PartOfSpeech.SYM,
            ["''"] = PartOfSpeech.PUNCT,
            ["\""] = PartOfSpeech.PUNCT,
            ["-LRB-"] = PartOfSpeech.PUNCT,
            ["-RRB-"] = PartOfSpeech.PUNCT,
            ["."] = PartOfSpeech.PUNCT,
            [","] = PartOfSpeech.PUNCT,
            [":"] = PartOfSpeech.PUNCT,
            ["AFX"] = PartOfSpeech.ADJ,
            ["CC"] = PartOfSpeech.CCONJ,
            ["CONJ"] = PartOfSpeech.CCONJ,
            ["CD"] = PartOfSpeech.NUM,
            ["DT"] = PartOfSpeech.DET,
            ["EX"] = PartOfSpeech.ADV,
            ["FW"] = PartOfSpeech.X,
            ["HYPH"] = PartOfSpeech.PUNCT,
            ["IN"] = PartOfSpeech.ADP,
            ["JJ"] = PartOfSpeech.ADJ,
            ["JJR"] = PartOfSpeech.ADJ,
            ["JJS"] = PartOfSpeech.ADJ,
            ["LS"] = PartOfSpeech.PUNCT,
            ["MD"] = PartOfSpeech.VERB,
            ["NIL"] = PartOfSpeech.X,
            ["NN"] = PartOfSpeech.NOUN,
            ["NNP"] = PartOfSpeech.PROPN,
            ["NNPS"] = PartOfSpeech.PROPN,
            ["NNS"] = PartOfSpeech.NOUN,
            ["PDT"] = PartOfSpeech.DET,
            ["POS"] = PartOfSpeech.PART,
            ["PRP"] = PartOfSpeech.PRON,
            ["PRP$"] = PartOfSpeech.DET,
            ["RB"] = PartOfSpeech.ADV,
            ["RBR"] = PartOfSpeech.ADV,
            ["RBS"] = PartOfSpeech.ADV,
            ["RP"] = PartOfSpeech.PART,
            ["PRT"] = PartOfSpeech.PART,

            ["SYM"] = PartOfSpeech.SYM,
            ["TO"] = PartOfSpeech.PART,
            ["UH"] = PartOfSpeech.INTJ,
            ["VERB"] = PartOfSpeech.VERB,
            ["VB"] = PartOfSpeech.VERB,
            ["VBD"] = PartOfSpeech.VERB,
            ["VBG"] = PartOfSpeech.VERB,
            ["VBN"] = PartOfSpeech.VERB,
            ["VBP"] = PartOfSpeech.VERB,
            ["VBZ"] = PartOfSpeech.VERB,
            ["WDT"] = PartOfSpeech.DET,
            ["WP"] = PartOfSpeech.PRON,
            ["WP$"] = PartOfSpeech.DET,
            ["WRB"] = PartOfSpeech.ADV,
            ["``"] = PartOfSpeech.PUNCT,
        };

        public static readonly HashSet<string> StringPOS = new HashSet<string>(Enum.GetValues(typeof(PartOfSpeech)).Cast<PartOfSpeech>().Select(e => e.ToString()));
    }
}