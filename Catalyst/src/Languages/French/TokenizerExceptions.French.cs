using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        private static object _lockFrenchExceptions = new object();
        private static Dictionary<int, TokenizationException> FrenchExceptions;

        private static Dictionary<int, TokenizationException> GetFrenchExceptions()
        {
            if (FrenchExceptions is null)
            {
                lock (_lockFrenchExceptions)
                {
                    if (FrenchExceptions is null)
                    {
                        FrenchExceptions = BaseExceptions();

                        Create(FrenchExceptions, "", "aujourd'hui|Aujourd'hui", "aujourd'hui|Aujourd'hui");
                        Create(FrenchExceptions, "", "J.-C.", "Jésus Christ");
                        Create(FrenchExceptions, "", "av.|janv.|févr.|avr.|juill.|sept.|oct.|nov.|déc.|apr.|Dr.|M.|Mr.|Mme.|Mlle.|n°|d°|St.|Ste.", "avant|janvier|février|avril|juillet|septembre|octobre|novembre|décembre|après|docteur|monsieur|monsieur|madame|mademoiselle|numéro|degrés|saint|sainte");

                        foreach (var (verb, verb_lemma) in new[] { ("a", "avoir"), ("est", "être"), ("semble", "sembler"), ("indique", "indiquer"), ("moque", "moquer"), ("passe", "passer") })
                        {
                            foreach (var pronoun in new[] { "elle", "il", "on" })
                            {
                                Create(FrenchExceptions, "", $"{verb}-t-{pronoun}", $"{verb_lemma} t {pronoun}");
                            }
                        }

                        Create(FrenchExceptions, "est|être", "-ce", "ce");
                        Create(FrenchExceptions, "", "qu'est-ce|n'est-ce", "que est ce|ne est ce");
                    }
                }
            }
            return FrenchExceptions;
        }
    }
}