using Catalyst.Models;
using System;
using System.Collections.Generic;
using System.Text;

namespace Catalyst
{
    public static partial class TokenizerExceptions
    {
        private static object _lockGermanExceptions = new object();
        private static Dictionary<int, TokenizationException> GermanExceptions;

        private static Dictionary<int, TokenizationException> GetGermanExceptions()
        {
            if (GermanExceptions is null)
            {
                lock (_lockGermanExceptions)
                {
                    if (GermanExceptions is null)
                    {
                        GermanExceptions = CreateBaseExceptions();

                        Create(GermanExceptions, "", "auf'm|du's|er's|hinter'm|ich's|ihr's|sie's|unter'm|vor'm|wir's|über'm", "auf dem|du es|er es|hinter dem|ich es|ihr es|sie es|unter dem|vor dem|wir es|über dem");
                        Create(GermanExceptions, "", "'S|'s|S'|s'|'n|'ne|'nen|'nem|Abb.|Abk.|Abt.|Apr.|Aug.|Bd.|Betr.|Bf.|Bhf.|Bsp.|Dez.|Di.|Do.|Fa.|Fam.|Feb.|Fr.|Frl.|Hbf.|Hr.|Hrn.|Jan.|Jh.|Jhd.|Jul.|Jun.|Mi.|Mio.|Mo.|Mrd.|Mrz.|MwSt.|Mär.|Nov.|Nr.|Okt.|Orig.|Pkt.|Prof.|Red.|Sa.|Sep.|Sept.|So.|Std.|Str.|Tel.|Tsd.|Univ.|abzgl.|allg.|bspw.|bzgl.|bzw.|d.h.|dgl.|ebd.|eigtl.|engl.|evtl.|frz.|gegr.|ggf.|ggfs.|ggü.|i.O.|i.d.R.|incl.|inkl.|insb.|kath.|lt.|max.|min.|mind.|mtl.|n.Chr.|orig.|röm.|s.o.|sog.|stellv.|tägl.|u.U.|u.s.w.|u.v.m.|usf.|usw.|uvm.|v.Chr.|v.a.|v.l.n.r.|vgl.|vllt.|vlt.|z.B.|z.Bsp.|z.T.|z.Z.|z.Zt.|z.b.|zzgl.|österr.", "'s|'s|'s|'s|ein|eine|einen|einem|Abbildung|Abkürzung|Abteilung|April|August|Band|Betreff|Bahnhof|Bahnhof|Beispiel|Dezember|Dienstag|Donnerstag|Firma|Familie|Februar|Frau|Fräulein|Hauptbahnhof|Herr|Herrn|Januar|Jahrhundert|Jahrhundert|Juli|Juni|Mittwoch|Million|Montag|Milliarde|März|Mehrwertsteuer|März|November|Nummer|Oktober|Original|Punkt|Professor|Redaktion|Samstag|September|September|Sonntag|Stunde|Straße|Telefon|Tausend|Universität|abzüglich|allgemein|beispielsweise|bezüglich|beziehungsweise|das heißt|dergleichen|ebenda|eigentlich|englisch|eventuell|französisch|gegründet|gegebenenfalls|gegebenenfalls|gegenüber|in Ordnung|in der Regel|inklusive|inklusive|insbesondere|katholisch|laut|maximal|minimal|mindestens|monatlich|nach Christus|original|römisch|siehe oben|so genannt|stellvertretend|täglich|unter Umständen|und so weiter|und vieles mehr|und so fort|und so weiter|und vieles mehr|vor Christus|vor allem|von links nach rechts|vergleiche|vielleicht|vielleicht|zum Beispiel|zum Beispiel|zum Teil|zur Zeit|zur Zeit|zum Beispiel|zuzüglich|österreichisch");
                        Create(GermanExceptions, "", "A.C.|a.D.|A.D.|A.G.|a.M.|a.Z.|Abs.|adv.|al.|B.A.|B.Sc.|betr.|biol.|Biol.|ca.|Chr.|Cie.|co.|Co.|D.C.|Dipl.-Ing.|Dipl.|Dr.|e.g.|e.V.|ehem.|entspr.|erm.|etc.|ev.|G.m.b.H.|geb.|Gebr.|gem.|h.c.|Hg.|hrsg.|Hrsg.|i.A.|i.e.|i.G.|i.Tr.|i.V.|Ing.|jr.|Jr.|jun.|jur.|K.O.|L.A.|lat.|M.A.|m.E.|m.M.|M.Sc.|Mr.|N.Y.|N.Y.C.|nat.|o.a.|o.ä.|o.g.|o.k.|O.K.|p.a.|p.s.|P.S.|pers.|phil.|q.e.d.|R.I.P.|rer.|sen.|St.|std.|u.a.|U.S.|U.S.A.|U.S.S.|Vol.|vs.|wiss.", "A.C.|a.D.|A.D.|A.G.|a.M.|a.Z.|Abs.|adv.|al.|B.A.|B.Sc.|betr.|biol.|Biol.|ca.|Chr.|Cie.|co.|Co.|D.C.|Dipl.-Ing.|Dipl.|Dr.|e.g.|e.V.|ehem.|entspr.|erm.|etc.|ev.|G.m.b.H.|geb.|Gebr.|gem.|h.c.|Hg.|hrsg.|Hrsg.|i.A.|i.e.|i.G.|i.Tr.|i.V.|Ing.|jr.|Jr.|jun.|jur.|K.O.|L.A.|lat.|M.A.|m.E.|m.M.|M.Sc.|Mr.|N.Y.|N.Y.C.|nat.|o.a.|o.ä.|o.g.|o.k.|O.K.|p.a.|p.s.|P.S.|pers.|phil.|q.e.d.|R.I.P.|rer.|sen.|St.|std.|u.a.|U.S.|U.S.A.|U.S.S.|Vol.|vs.|wiss.");
                    }
                }
            }
            return GermanExceptions;
        }
    }
}