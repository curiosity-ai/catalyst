
using System;
using System.Collections.Generic;
using Catalyst;
using Mosaik.Core;

namespace Catalyst.Models
{
    public static partial class German
    {
        internal sealed class TokenizerExceptions 
        {
            internal static Dictionary<int, TokenizationException> Get()
            {
                var exceptions = Catalyst.TokenizerExceptions.CreateBaseExceptions();

                Catalyst.TokenizerExceptions.Create(exceptions, "", "auf'm|du's|er's|hinter'm|ich's|ihr's|sie's|unter'm|vor'm|wir's|�ber'm", "auf dem|du es|er es|hinter dem|ich es|ihr es|sie es|unter dem|vor dem|wir es|�ber dem");
                Catalyst.TokenizerExceptions.Create(exceptions, "", "'S|'s|S'|s'|'n|'ne|'nen|'nem|Abb.|Abk.|Abt.|Apr.|Aug.|Bd.|Betr.|Bf.|Bhf.|Bsp.|Dez.|Di.|Do.|Fa.|Fam.|Feb.|Fr.|Frl.|Hbf.|Hr.|Hrn.|Jan.|Jh.|Jhd.|Jul.|Jun.|Mi.|Mio.|Mo.|Mrd.|Mrz.|MwSt.|M�r.|Nov.|Nr.|Okt.|Orig.|Pkt.|Prof.|Red.|Sa.|Sep.|Sept.|So.|Std.|Str.|Tel.|Tsd.|Univ.|abzgl.|allg.|bspw.|bzgl.|bzw.|d.h.|dgl.|ebd.|eigtl.|engl.|evtl.|frz.|gegr.|ggf.|ggfs.|gg�.|i.O.|i.d.R.|incl.|inkl.|insb.|kath.|lt.|max.|min.|mind.|mtl.|n.Chr.|orig.|r�m.|s.o.|sog.|stellv.|t�gl.|u.U.|u.s.w.|u.v.m.|usf.|usw.|uvm.|v.Chr.|v.a.|v.l.n.r.|vgl.|vllt.|vlt.|z.B.|z.Bsp.|z.T.|z.Z.|z.Zt.|z.b.|zzgl.|�sterr.", "'s|'s|'s|'s|ein|eine|einen|einem|Abbildung|Abk�rzung|Abteilung|April|August|Band|Betreff|Bahnhof|Bahnhof|Beispiel|Dezember|Dienstag|Donnerstag|Firma|Familie|Februar|Frau|Fr�ulein|Hauptbahnhof|Herr|Herrn|Januar|Jahrhundert|Jahrhundert|Juli|Juni|Mittwoch|Million|Montag|Milliarde|M�rz|Mehrwertsteuer|M�rz|November|Nummer|Oktober|Original|Punkt|Professor|Redaktion|Samstag|September|September|Sonntag|Stunde|Stra�e|Telefon|Tausend|Universit�t|abz�glich|allgemein|beispielsweise|bez�glich|beziehungsweise|das hei�t|dergleichen|ebenda|eigentlich|englisch|eventuell|franz�sisch|gegr�ndet|gegebenenfalls|gegebenenfalls|gegen�ber|in Ordnung|in der Regel|inklusive|inklusive|insbesondere|katholisch|laut|maximal|minimal|mindestens|monatlich|nach Christus|original|r�misch|siehe oben|so genannt|stellvertretend|t�glich|unter Umst�nden|und so weiter|und vieles mehr|und so fort|und so weiter|und vieles mehr|vor Christus|vor allem|von links nach rechts|vergleiche|vielleicht|vielleicht|zum Beispiel|zum Beispiel|zum Teil|zur Zeit|zur Zeit|zum Beispiel|zuz�glich|�sterreichisch");
                Catalyst.TokenizerExceptions.Create(exceptions, "", "A.C.|a.D.|A.D.|A.G.|a.M.|a.Z.|Abs.|adv.|al.|B.A.|B.Sc.|betr.|biol.|Biol.|ca.|Chr.|Cie.|co.|Co.|D.C.|Dipl.-Ing.|Dipl.|Dr.|e.g.|e.V.|ehem.|entspr.|erm.|etc.|ev.|G.m.b.H.|geb.|Gebr.|gem.|h.c.|Hg.|hrsg.|Hrsg.|i.A.|i.e.|i.G.|i.Tr.|i.V.|Ing.|jr.|Jr.|jun.|jur.|K.O.|L.A.|lat.|M.A.|m.E.|m.M.|M.Sc.|Mr.|N.Y.|N.Y.C.|nat.|o.a.|o.�.|o.g.|o.k.|O.K.|p.a.|p.s.|P.S.|pers.|phil.|q.e.d.|R.I.P.|rer.|sen.|St.|std.|u.a.|U.S.|U.S.A.|U.S.S.|Vol.|vs.|wiss.", "A.C.|a.D.|A.D.|A.G.|a.M.|a.Z.|Abs.|adv.|al.|B.A.|B.Sc.|betr.|biol.|Biol.|ca.|Chr.|Cie.|co.|Co.|D.C.|Dipl.-Ing.|Dipl.|Dr.|e.g.|e.V.|ehem.|entspr.|erm.|etc.|ev.|G.m.b.H.|geb.|Gebr.|gem.|h.c.|Hg.|hrsg.|Hrsg.|i.A.|i.e.|i.G.|i.Tr.|i.V.|Ing.|jr.|Jr.|jun.|jur.|K.O.|L.A.|lat.|M.A.|m.E.|m.M.|M.Sc.|Mr.|N.Y.|N.Y.C.|nat.|o.a.|o.�.|o.g.|o.k.|O.K.|p.a.|p.s.|P.S.|pers.|phil.|q.e.d.|R.I.P.|rer.|sen.|St.|std.|u.a.|U.S.|U.S.A.|U.S.S.|Vol.|vs.|wiss.");

                return exceptions;
            }
        }
    }
}
