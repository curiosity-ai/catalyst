using UID;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Collections.Concurrent;

namespace Catalyst
{
    public static class CharSpanExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsNullOrWhiteSpace(this ReadOnlySpan<char> span)    { if (span.Length == 0) { return true;  }  for (int i = 0; i < span.Length; i++) { if (!char.IsWhiteSpace(span[i]))                                       { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsCapitalized(this ReadOnlySpan<char> span)         { if (span.Length == 0) { return false; }  return char.IsUpper(span[0]) && !span.IsAllUpperCase(); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsAllUpperCase(this ReadOnlySpan<char> span)        { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (char.IsLower(span[i]) && !char.IsNumber(span[i]))                  { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsAllLowerCase(this ReadOnlySpan<char> span)        { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (char.IsUpper(span[i]) && !char.IsNumber(span[i]))                  { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsAllLetterOrDigit(this ReadOnlySpan<char> span)    { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!char.IsLetterOrDigit(span[i]))                                    { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsLetter(this ReadOnlySpan<char> span)              { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!char.IsLetter(span[i]))                                           { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsDigit(this ReadOnlySpan<char> span)               { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!char.IsDigit(span[i]))                                            { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsNumeric(this ReadOnlySpan<char> span)             { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.NumericCharacters.Contains(span[i]))             { return false; } } return CharacterClasses.RE_AllNumeric.IsMatch(new string(span.ToArray())); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool HasNumeric(this ReadOnlySpan<char> span)            { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (CharacterClasses.NumericCharacters.Contains(span[i]))              { return true;  } } return false; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsSentencePunctuation(this ReadOnlySpan<char> span) { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.SentencePunctuationCharacters.Contains(span[i])) { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsAnyPunctuation(this ReadOnlySpan<char> span)      { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (char.IsPunctuation(span[i]))                                       { return true;  } } return false; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsPunctuation(this ReadOnlySpan<char> span)         { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!char.IsPunctuation(span[i]))                                      { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsSymbol(this ReadOnlySpan<char> span)              { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!char.IsSymbol(span[i]))                                           { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsCurrency(this ReadOnlySpan<char> span)            { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.CurrencyCharacters.Contains(span[i]))            { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsHyphen(this ReadOnlySpan<char> span)              { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.HyphenCharacters.Contains(span[i]))              { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsOpenQuote(this ReadOnlySpan<char> span)           { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.OpenQuotesCharacters.Contains(span[i]))          { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsEmailLocalPart(this ReadOnlySpan<char> span)      { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.EmailLocalPartCharacters.Contains(span[i]))      { return false; } } return true;  }
        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsEmailDomainPart(this ReadOnlySpan<char> span)     { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.EmailDomainPartCharacters.Contains(span[i]))     { return false; } } return true;  }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool IsQuoteCharacters(this ReadOnlySpan<char> span)     { if (span.Length == 0) { return false; } for (int i = 0; i < span.Length; i++) { if (!CharacterClasses.QuotesCharacters.Contains(span[i])) { return false; } } return true; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)] public static bool HasLigatures(this ReadOnlySpan<char> span)          { if (span.Length == 0) { return false; }  for (int i = 0; i < span.Length; i++) { if(span[i] < CharacterClasses.MinimumLigatureValue || span[i] > CharacterClasses.MaximumLigatureValue) { continue; } else { if (CharacterClasses.UnicodeLatinLigatures.Contains(span[i])) { return true; } } } return false; }

        //TODO: Split these two in URL and Email
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsLikeURLorEmail(this ReadOnlySpan<char> candidate)
        {
            bool value = candidate.IsLikeURLorEmailInternal();
            if(value)
            {
                // check that candidate doesn't contain invalid url characters
                for(int i = 0; i < candidate.Length; i++)
                {
                    if (!CharacterClasses.ValidURLCharacters.Contains(candidate[i])) { return false; }
                }
            }
            return value;
        }

        public static bool IsEmoji(this ReadOnlySpan<char> candidate, out int count)
        {
            //approximate check for membership of the following table: http://unicode.org/Public/emoji/11.0/emoji-test.txt
            //Does not check for valid emoji for multi-char emojis, just assumes if each part of the sequence of chars is part of the set, then the final combination is an emoji

            //Auto-generated code using EmojiToCharClass project

            if (candidate.Length > 0)
            {
                var a = (ushort)candidate[0];

                //Shortcut for first test
                if ( (a < 35 || (a > 57)) && a < 8252 && a != 169 && a != 174)
                {
                    count = 0;
                    return false;
                }

                if ((a == 55357) || (a == 55358) || (a == 9786) || (a == 9785) || (a == 9760) || (a == 55356) || (a == 9975) || (a == 9977) || (a == 9757) || (a == 9996) || (a == 9995) || (a == 9994) || (a == 9997) || (a == 10084) || (a == 10083) || (a == 9937) || (a == 9752) || (a == 9749) || (a == 9968) || (a == 9962) || (a == 9961) || (a == 9970) || (a == 9978) || (a == 9832) || (a == 9981) || (a == 9875) || (a == 9973) || (a == 9972) || (a == 9992) || (a == 8987) || (a == 9203) || (a == 8986) || (a == 9200) || (a == 9201) || (a == 9202) || (a == 9728) || (a == 11088) || (a == 9729) || (a == 9925) || (a == 9928) || (a == 9730) || (a == 9748) || (a == 9969) || (a == 9889) || (a == 10052) || (a == 9731) || (a == 9924) || (a == 9732) || (a == 10024) || (a == 9917) || (a == 9918) || (a == 9971) || (a == 9976) || (a == 9824) || (a == 9829) || (a == 9830) || (a == 9827) || (a == 9742) || (a == 9000) || (a == 9993) || (a == 9999) || (a == 10002) || (a == 9986) || (a == 9935) || (a == 9874) || (a == 9876) || (a == 9881) || (a == 9879) || (a == 9878) || (a == 9939) || (a == 9904) || (a == 9905) || (a == 9855) || (a == 9888) || (a == 9940) || (a == 9762) || (a == 9763) || (a == 11014) || (a == 8599) || (a == 10145) || (a == 8600) || (a == 11015) || (a == 8601) || (a == 11013) || (a == 8598) || (a == 8597) || (a == 8596) || (a == 8617) || (a == 8618) || (a == 10548) || (a == 10549) || (a == 9883) || (a == 10017) || (a == 9784) || (a == 9775) || (a == 10013) || (a == 9766) || (a == 9770) || (a == 9774) || (a == 9800) || (a == 9801) || (a == 9802) || (a == 9803) || (a == 9804) || (a == 9805) || (a == 9806) || (a == 9807) || (a == 9808) || (a == 9809) || (a == 9810) || (a == 9811) || (a == 9934) || (a == 9654) || (a == 9193) || (a == 9197) || (a == 9199) || (a == 9664) || (a == 9194) || (a == 9198) || (a == 9195) || (a == 9196) || (a == 9208) || (a == 9209) || (a == 9210) || (a == 9167) || (a == 9792) || (a == 9794) || (a == 9877) || (a == 9851) || (a == 9884) || (a == 11093) || (a == 9989) || (a == 9745) || (a == 10004) || (a == 10006) || (a == 10060) || (a == 10062) || (a == 10133) || (a == 10134) || (a == 10135) || (a == 10160) || (a == 10175) || (a == 12349) || (a == 10035) || (a == 10036) || (a == 10055) || (a == 8252) || (a == 8265) || (a == 10067) || (a == 10068) || (a == 10069) || (a == 10071) || (a == 12336) || (a == 169) || (a == 174) || (a == 8482) || (a == 35) || (a == 42) || (a == 48) || (a == 49) || (a == 50) || (a == 51) || (a == 52) || (a == 53) || (a == 54) || (a == 55) || (a == 56) || (a == 57) || (a == 8505) || (a == 9410) || (a == 12951) || (a == 12953) || (a == 9642) || (a == 9643) || (a == 9723) || (a == 9724) || (a == 9725) || (a == 9726) || (a == 11035) || (a == 11036) || (a == 9898) || (a == 9899))
                {
                    if (candidate.Length > 1)
                    {
                        var b = (ushort)candidate[1];
                        if ((b == 56832) || (b == 56833) || (b == 56834) || (b == 56611) || (b == 56835) || (b == 56836) || (b == 56837) || (b == 56838) || (b == 56841) || (b == 56842) || (b == 56843) || (b == 56846) || (b == 56845) || (b == 56856) || (b == 56855) || (b == 56857) || (b == 56858) || (b == 65039) || (b == 56898) || (b == 56599) || (b == 56617) || (b == 56596) || (b == 56616) || (b == 56848) || (b == 56849) || (b == 56886) || (b == 56900) || (b == 56847) || (b == 56867) || (b == 56869) || (b == 56878) || (b == 56592) || (b == 56879) || (b == 56874) || (b == 56875) || (b == 56884) || (b == 56844) || (b == 56859) || (b == 56860) || (b == 56861) || (b == 56612) || (b == 56850) || (b == 56851) || (b == 56852) || (b == 56853) || (b == 56899) || (b == 56593) || (b == 56882) || (b == 56897) || (b == 56854) || (b == 56862) || (b == 56863) || (b == 56868) || (b == 56866) || (b == 56877) || (b == 56870) || (b == 56871) || (b == 56872) || (b == 56873) || (b == 56623) || (b == 56876) || (b == 56880) || (b == 56881) || (b == 56883) || (b == 56618) || (b == 56885) || (b == 56865) || (b == 56864) || (b == 56620) || (b == 56887) || (b == 56594) || (b == 56597) || (b == 56610) || (b == 56622) || (b == 56615) || (b == 56839) || (b == 56608) || (b == 56609) || (b == 56613) || (b == 56619) || (b == 56621) || (b == 56784) || (b == 56595) || (b == 56840) || (b == 56447) || (b == 56441) || (b == 56442) || (b == 56448) || (b == 56443) || (b == 56445) || (b == 56446) || (b == 56598) || (b == 56489) || (b == 56890) || (b == 56888) || (b == 56889) || (b == 56891) || (b == 56892) || (b == 56893) || (b == 56896) || (b == 56895) || (b == 56894) || (b == 56904) || (b == 56905) || (b == 56906) || (b == 56438) || (b == 56786) || (b == 56422) || (b == 56423) || (b == 56785) || (b == 56424) || (b == 56425) || (b == 56787) || (b == 56436) || (b == 56437) || (b == 56430) || (b == 56693) || (b == 56450) || (b == 56439) || (b == 56628) || (b == 56440) || (b == 56435) || (b == 56434) || (b == 56789) || (b == 56788) || (b == 56433) || (b == 56629) || (b == 56432) || (b == 56624) || (b == 56625) || (b == 56444) || (b == 57221) || (b == 56630) || (b == 56793) || (b == 56794) || (b == 56795) || (b == 56796) || (b == 56797) || (b == 56798) || (b == 56799) || (b == 56909) || (b == 56910) || (b == 56901) || (b == 56902) || (b == 56449) || (b == 56907) || (b == 56903) || (b == 56614) || (b == 56631) || (b == 56454) || (b == 56455) || (b == 57014) || (b == 57283) || (b == 56451) || (b == 56698) || (b == 56431) || (b == 56790) || (b == 56791) || (b == 56792) || (b == 57024) || (b == 57036) || (b == 56692) || (b == 56803) || (b == 56420) || (b == 56421) || (b == 56634) || (b == 57287) || (b == 57282) || (b == 57292) || (b == 57284) || (b == 56995) || (b == 57290) || (b == 55356) || (b == 57291) || (b == 57012) || (b == 57013) || (b == 57294) || (b == 57293) || (b == 56632) || (b == 56636) || (b == 56637) || (b == 56638) || (b == 56633) || (b == 56427) || (b == 56428) || (b == 56429) || (b == 56463) || (b == 56465) || (b == 56426) || (b == 56627) || (b == 56490) || (b == 56392) || (b == 56393) || (b == 56390) || (b == 56725) || (b == 56391) || (b == 56606) || (b == 56726) || (b == 56600) || (b == 56601) || (b == 56720) || (b == 56396) || (b == 56397) || (b == 56398) || (b == 56394) || (b == 56603) || (b == 56604) || (b == 56602) || (b == 56395) || (b == 56607) || (b == 56399) || (b == 56400) || (b == 56908) || (b == 56626) || (b == 56911) || (b == 56605) || (b == 56453) || (b == 56386) || (b == 56387) || (b == 56419) || (b == 56384) || (b == 56385) || (b == 56800) || (b == 56389) || (b == 56388) || (b == 56459) || (b == 56472) || (b == 56467) || (b == 56468) || (b == 56469) || (b == 56470) || (b == 56471) || (b == 56473) || (b == 56474) || (b == 56475) || (b == 56801) || (b == 56476) || (b == 56740) || (b == 56477) || (b == 56478) || (b == 56479) || (b == 56460) || (b == 56484) || (b == 56482) || (b == 56483) || (b == 56485) || (b == 56486) || (b == 56488) || (b == 56491) || (b == 56492) || (b == 56808) || (b == 56815) || (b == 56493) || (b == 56691) || (b == 56403) || (b == 56694) || (b == 56404) || (b == 56405) || (b == 56406) || (b == 56804) || (b == 56805) || (b == 56806) || (b == 56407) || (b == 56408) || (b == 56409) || (b == 56410) || (b == 56411) || (b == 56412) || (b == 56413) || (b == 57037) || (b == 57234) || (b == 56414) || (b == 56415) || (b == 56416) || (b == 56417) || (b == 56418) || (b == 56401) || (b == 56402) || (b == 57257) || (b == 57235) || (b == 56802) || (b == 56575) || (b == 56452) || (b == 56461) || (b == 56462) || (b == 56373) || (b == 56338) || (b == 56717) || (b == 56374) || (b == 56341) || (b == 56361) || (b == 56378) || (b == 56714) || (b == 56369) || (b == 56328) || (b == 56705) || (b == 56367) || (b == 56325) || (b == 56326) || (b == 56372) || (b == 56334) || (b == 56708) || (b == 56723) || (b == 56716) || (b == 56366) || (b == 56322) || (b == 56323) || (b == 56324) || (b == 56375) || (b == 56342) || (b == 56343) || (b == 56381) || (b == 56335) || (b == 56337) || (b == 56336) || (b == 56362) || (b == 56363) || (b == 56722) || (b == 56344) || (b == 56719) || (b == 56365) || (b == 56321) || (b == 56320) || (b == 56377) || (b == 56368) || (b == 56327) || (b == 56383) || (b == 56724) || (b == 56711) || (b == 56379) || (b == 56360) || (b == 56380) || (b == 56382) || (b == 56707) || (b == 56340) || (b == 56339) || (b == 56355) || (b == 56356) || (b == 56357) || (b == 56358) || (b == 56359) || (b == 56650) || (b == 56709) || (b == 56710) || (b == 56713) || (b == 56376) || (b == 56330) || (b == 56354) || (b == 56718) || (b == 56333) || (b == 56370) || (b == 56329) || (b == 56371) || (b == 56331) || (b == 56364) || (b == 56351) || (b == 56352) || (b == 56353) || (b == 56712) || (b == 56345) || (b == 56346) || (b == 56704) || (b == 56721) || (b == 56332) || (b == 56715) || (b == 56347) || (b == 56348) || (b == 56349) || (b == 56350) || (b == 56727) || (b == 56695) || (b == 56696) || (b == 56706) || (b == 56464) || (b == 57144) || (b == 56494) || (b == 57333) || (b == 57145) || (b == 56640) || (b == 57146) || (b == 57147) || (b == 57148) || (b == 57143) || (b == 57137) || (b == 57138) || (b == 57139) || (b == 57140) || (b == 57141) || (b == 57150) || (b == 57151) || (b == 57152) || (b == 57153) || (b == 57154) || (b == 57155) || (b == 57159) || (b == 57160) || (b == 57161) || (b == 57162) || (b == 57163) || (b == 57164) || (b == 57165) || (b == 57166) || (b == 57167) || (b == 57168) || (b == 57169) || (b == 57170) || (b == 57171) || (b == 56669) || (b == 57157) || (b == 56677) || (b == 56657) || (b == 57158) || (b == 56660) || (b == 56661) || (b == 57149) || (b == 57142) || (b == 56658) || (b == 56678) || (b == 57156) || (b == 56668) || (b == 57136) || (b == 57182) || (b == 56656) || (b == 56662) || (b == 56680) || (b == 56670) || (b == 56768) || (b == 57174) || (b == 57175) || (b == 56681) || (b == 56659) || (b == 57172) || (b == 57183) || (b == 57173) || (b == 57133) || (b == 56682) || (b == 57134) || (b == 57135) || (b == 56665) || (b == 56666) || (b == 57203) || (b == 56664) || (b == 57202) || (b == 56675) || (b == 56663) || (b == 57215) || (b == 56683) || (b == 57201) || (b == 57176) || (b == 57177) || (b == 57178) || (b == 57179) || (b == 57180) || (b == 57181) || (b == 57184) || (b == 57186) || (b == 57187) || (b == 57188) || (b == 57189) || (b == 57185) || (b == 56671) || (b == 56672) || (b == 56673) || (b == 57190) || (b == 57191) || (b == 57192) || (b == 57193) || (b == 57194) || (b == 57218) || (b == 57200) || (b == 56679) || (b == 57195) || (b == 57196) || (b == 57197) || (b == 57198) || (b == 57199) || (b == 57212) || (b == 56667) || (b == 57205) || (b == 57206) || (b == 57214) || (b == 57207) || (b == 57208) || (b == 57209) || (b == 57210) || (b == 57211) || (b == 56642) || (b == 56643) || (b == 56676) || (b == 56674) || (b == 57213) || (b == 57204) || (b == 56644) || (b == 57338) || (b == 57101) || (b == 57102) || (b == 57103) || (b == 57104) || (b == 56826) || (b == 56830) || (b == 57300) || (b == 57099) || (b == 56827) || (b == 57301) || (b == 57302) || (b == 57308) || (b == 57309) || (b == 57310) || (b == 57311) || (b == 57307) || (b == 57303) || (b == 57304) || (b == 57305) || (b == 57306) || (b == 57313) || (b == 57314) || (b == 57315) || (b == 57316) || (b == 57317) || (b == 57318) || (b == 57320) || (b == 57321) || (b == 57322) || (b == 57323) || (b == 57324) || (b == 57325) || (b == 57327) || (b == 57328) || (b == 56466) || (b == 56828) || (b == 56829) || (b == 56652) || (b == 56653) || (b == 56651) || (b == 57089) || (b == 57091) || (b == 57092) || (b == 57093) || (b == 57094) || (b == 57095) || (b == 57097) || (b == 57100) || (b == 57248) || (b == 57249) || (b == 57250) || (b == 56456) || (b == 57258) || (b == 57261) || (b == 56764) || (b == 57256) || (b == 57264) || (b == 56962) || (b == 56963) || (b == 56964) || (b == 56965) || (b == 56966) || (b == 56967) || (b == 56968) || (b == 56969) || (b == 56970) || (b == 56989) || (b == 56990) || (b == 56971) || (b == 56972) || (b == 56973) || (b == 56974) || (b == 56976) || (b == 56977) || (b == 56978) || (b == 56979) || (b == 56980) || (b == 56981) || (b == 56982) || (b == 56983) || (b == 56984) || (b == 56985) || (b == 56986) || (b == 56987) || (b == 56988) || (b == 57010) || (b == 57076) || (b == 57077) || (b == 56975) || (b == 57059) || (b == 57060) || (b == 57000) || (b == 56997) || (b == 56998) || (b == 56999) || (b == 57041) || (b == 57078) || (b == 56996) || (b == 57075) || (b == 57061) || (b == 56994) || (b == 57065) || (b == 57067) || (b == 57068) || (b == 56506) || (b == 56961) || (b == 56991) || (b == 56992) || (b == 56993) || (b == 57072) || (b == 56960) || (b == 57080) || (b == 57038) || (b == 57002) || (b == 57039) || (b == 57035) || (b == 57021) || (b == 57023) || (b == 57025) || (b == 56688) || (b == 57105) || (b == 57106) || (b == 57107) || (b == 57108) || (b == 57109) || (b == 57110) || (b == 57111) || (b == 57112) || (b == 57113) || (b == 57114) || (b == 57115) || (b == 57116) || (b == 57121) || (b == 57117) || (b == 57118) || (b == 57119) || (b == 57120) || (b == 57124) || (b == 57125) || (b == 57126) || (b == 57127) || (b == 57128) || (b == 57129) || (b == 57130) || (b == 57131) || (b == 57132) || (b == 57088) || (b == 57096) || (b == 57090) || (b == 56487) || (b == 57098) || (b == 57219) || (b == 57220) || (b == 57222) || (b == 57223) || (b == 57224) || (b == 57225) || (b == 57226) || (b == 57227) || (b == 57229) || (b == 57230) || (b == 57231) || (b == 57232) || (b == 57233) || (b == 57216) || (b == 57217) || (b == 57239) || (b == 57247) || (b == 57259) || (b == 57238) || (b == 57286) || (b == 57285) || (b == 56647) || (b == 56648) || (b == 56649) || (b == 57280) || (b == 57296) || (b == 57288) || (b == 57289) || (b == 57278) || (b == 57265) || (b == 57267) || (b == 57297) || (b == 57298) || (b == 57299) || (b == 57336) || (b == 56645) || (b == 57263) || (b == 57251) || (b == 57277) || (b == 57279) || (b == 57079) || (b == 57262) || (b == 56697) || (b == 57266) || (b == 56527) || (b == 57268) || (b == 56583) || (b == 56584) || (b == 56585) || (b == 56586) || (b == 56546) || (b == 56547) || (b == 56559) || (b == 57276) || (b == 57269) || (b == 57270) || (b == 57241) || (b == 57242) || (b == 57243) || (b == 57252) || (b == 57255) || (b == 56571) || (b == 57271) || (b == 57272) || (b == 57273) || (b == 57274) || (b == 57275) || (b == 56641) || (b == 56561) || (b == 56562) || (b == 56542) || (b == 56543) || (b == 56544) || (b == 56587) || (b == 56588) || (b == 56507) || (b == 56741) || (b == 56744) || (b == 56753) || (b == 56754) || (b == 56509) || (b == 56510) || (b == 56511) || (b == 56512) || (b == 57253) || (b == 57246) || (b == 56573) || (b == 57260) || (b == 56570) || (b == 56567) || (b == 56568) || (b == 56569) || (b == 56572) || (b == 56589) || (b == 56590) || (b == 56545) || (b == 56687) || (b == 56481) || (b == 57326) || (b == 56532) || (b == 56533) || (b == 56534) || (b == 56535) || (b == 56536) || (b == 56537) || (b == 56538) || (b == 56531) || (b == 56530) || (b == 56515) || (b == 56540) || (b == 56516) || (b == 56560) || (b == 56529) || (b == 57335) || (b == 56496) || (b == 56500) || (b == 56501) || (b == 56502) || (b == 56503) || (b == 56504) || (b == 56499) || (b == 56505) || (b == 56497) || (b == 56498) || (b == 56551) || (b == 56552) || (b == 56553) || (b == 56548) || (b == 56549) || (b == 56550) || (b == 56555) || (b == 56554) || (b == 56556) || (b == 56557) || (b == 56558) || (b == 56819) || (b == 56541) || (b == 56508) || (b == 56513) || (b == 56514) || (b == 56770) || (b == 56517) || (b == 56518) || (b == 56519) || (b == 56520) || (b == 56521) || (b == 56522) || (b == 56523) || (b == 56524) || (b == 56525) || (b == 56526) || (b == 56528) || (b == 56771) || (b == 56772) || (b == 56591) || (b == 57056) || (b == 57337) || (b == 57057) || (b == 56457) || (b == 56458) || (b == 57004) || (b == 56831) || (b == 57058) || (b == 57042) || (b == 57319) || (b == 57006) || (b == 57008) || (b == 57017) || (b == 57018) || (b == 57019) || (b == 57020) || (b == 57022) || (b == 57026) || (b == 57027) || (b == 57028) || (b == 57029) || (b == 57016) || (b == 57003) || (b == 57011) || (b == 57005) || (b == 57007) || (b == 57009) || (b == 57015) || (b == 56565) || (b == 56579) || (b == 56580) || (b == 57040) || (b == 56654) || (b == 56576) || (b == 56577) || (b == 56578) || (b == 57254) || (b == 56581) || (b == 56582) || (b == 56566) || (b == 56563) || (b == 56564) || (b == 56539) || (b == 56495) || (b == 56689) || (b == 56702) || (b == 56703) || (b == 56728) || (b == 56730) || (b == 56912) || (b == 56913) || (b == 56635) || (b == 56480) || (b == 57281) || (b == 57001) || (b == 57228) || (b == 57332) || (b == 57331) || (b == 56807) || (b == 56809) || (b == 56810) || (b == 56811) || (b == 56812) || (b == 56813) || (b == 56814) || (b == 56816) || (b == 56817) || (b == 56818) || (b == 56820) || (b == 56821) || (b == 56822) || (b == 56823) || (b == 56824) || (b == 56825))
                        {
                            if (candidate.Length > 2)
                            {
                                var c = (ushort)candidate[2];
                                if ((c == 55356) || (c == 8205) || (c == 65039) || (c == 57339) || (c == 57340) || (c == 57341) || (c == 57342) || (c == 57343) || (c == 8419) || (c == 56128))
                                {
                                    if (candidate.Length > 3)
                                    {
                                        var d = (ushort)candidate[3];
                                        if ((d == 57339) || (d == 57340) || (d == 57341) || (d == 57342) || (d == 57343) || (d == 9877) || (d == 55356) || (d == 9878) || (d == 55357) || (d == 9992) || (d == 9794) || (d == 9792) || (d == 8205) || (d == 10084) || (d == 56808) || (d == 56809) || (d == 56810) || (d == 56811) || (d == 56812) || (d == 56814) || (d == 56817) || (d == 56818) || (d == 56820) || (d == 56822) || (d == 56823) || (d == 56824) || (d == 56825) || (d == 56826) || (d == 56828) || (d == 56829) || (d == 56831) || (d == 56806) || (d == 56807) || (d == 56813) || (d == 56815) || (d == 56819) || (d == 56827) || (d == 56830) || (d == 56816) || (d == 56821) || (d == 56423))
                                        {
                                            if (candidate.Length > 4)
                                            {
                                                var e = (ushort)candidate[4];
                                                if ((e == 65039) || (e == 8205) || (e == 57235) || (e == 57323) || (e == 57150) || (e == 57203) || (e == 56615) || (e == 57325) || (e == 56508) || (e == 56620) || (e == 56507) || (e == 57252) || (e == 57256) || (e == 56960) || (e == 56978) || (e == 9794) || (e == 9792) || (e == 56425) || (e == 56424) || (e == 56422) || (e == 56423) || (e == 55357) || (e == 55356) || (e == 56128))
                                                {
                                                    if (candidate.Length > 5)
                                                    {
                                                        var f = (ushort)candidate[5];
                                                        if ((f == 9877) || (f == 55356) || (f == 9878) || (f == 55357) || (f == 9992) || (f == 9794) || (f == 9792) || (f == 65039) || (f == 8205) || (f == 56808) || (f == 57096) || (f == 56418))
                                                        {
                                                            if (candidate.Length > 6)
                                                            {
                                                                var g = (ushort)candidate[6];
                                                                if ((g == 65039) || (g == 57235) || (g == 57323) || (g == 57150) || (g == 57203) || (g == 56615) || (g == 57325) || (g == 56508) || (g == 56620) || (g == 56507) || (g == 57252) || (g == 57256) || (g == 56960) || (g == 56978) || (g == 55357) || (g == 56128))
                                                                {
                                                                    if (candidate.Length > 7)
                                                                    {
                                                                        var h = (ushort)candidate[7];
                                                                        if ((h == 56459) || (h == 56424) || (h == 56425) || (h == 56422) || (h == 56423) || (h == 56421) || (h == 56435) || (h == 56439))
                                                                        {
                                                                            if (candidate.Length > 8)
                                                                            {
                                                                                var i = (ushort)candidate[8];
                                                                                if ((i == 8205) || (i == 56128))
                                                                                {
                                                                                    if (candidate.Length > 9)
                                                                                    {
                                                                                        var j = (ushort)candidate[9];
                                                                                        if ((j == 55357) || (j == 56430) || (j == 56419) || (j == 56428))
                                                                                        {
                                                                                            if (candidate.Length > 10)
                                                                                            {
                                                                                                var k = (ushort)candidate[10];
                                                                                                if ((k == 56424) || (k == 56425) || (k == 56422) || (k == 56423) || (k == 56128))
                                                                                                {
                                                                                                    if (candidate.Length > 11)
                                                                                                    {
                                                                                                        var l = (ushort)candidate[11];
                                                                                                        if ((l == 56423) || (l == 56436) || (l == 56435))
                                                                                                        {
                                                                                                            if (candidate.Length > 12)
                                                                                                            {
                                                                                                                var m = (ushort)candidate[12];
                                                                                                                if ((m == 56128))
                                                                                                                {
                                                                                                                    if (candidate.Length > 13)
                                                                                                                    {
                                                                                                                        var n = (ushort)candidate[13];
                                                                                                                        if ((n == 56447))
                                                                                                                        {
                                                                                                                            count = 14; return true;
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                            count = 12; return true;
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                            count = 10; return true;
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                            count = 8; return true;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                            count = 6; return true;
                                                        }
                                                    }
                                                }
                                            }
                                            count = 4; return true;
                                        }
                                    }
                                }
                            }
                            count = 2; return true;
                        }
                    }
                }
            }

            count = 0;
            return false;
        }


        //TODO: Check what's the best option:https://mathiasbynens.be/demo/url-regex
        private static Regex RE_IsURL = new Regex(@"^(\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'"".,<>?«»“”‘’])))$", RegexOptions.Compiled & RegexOptions.CultureInvariant & RegexOptions.IgnoreCase, TimeSpan.FromMilliseconds(10)); //Timeout in 10 millisecond!
        //private static Regex RE_IsEmail = new Regex(@"^\w+([-+.']\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$", RegexOptions.Compiled & RegexOptions.CultureInvariant & RegexOptions.IgnoreCase, TimeSpan.FromMilliseconds(1));

        private static bool IsURLRegex(ReadOnlySpan<char> candidate)
        {
            try
            {
                return RE_IsURL.IsMatch(candidate.ToString());
            }
            catch (RegexMatchTimeoutException ex)
            {
                return false;
            }
            catch
            {
                throw;
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsLikeURLorEmailInternal(this ReadOnlySpan<char> candidate)
        {
            if (candidate.Length < 5) { return false; }
            int L2 = candidate.Length - 2;

            int dotCount = 0;
            for (int i = 2; i < L2; i++)
            {
                if (candidate[i] == ':' && candidate[i + 1] == '/' && candidate[i + 2] == '/')
                {
                    // contains '://'
                    return IsURLRegex(candidate);
                }
                else if (candidate[i] == '@')
                {
                    if(i > 8 && candidate.StartsWith("mailto:".AsSpan(), StringComparison.InvariantCultureIgnoreCase))
                    {
                        //For the case starting with mailto:, skip the mailto part
                        //This does not attempt to validade anything - i.e. it will match invalid emails
                        if (candidate.Slice(7, i - 1 - 6).IsEmailLocalPart() && candidate.Slice(i + 1).IsEmailDomainPart())
                        {
                            return true;
                        }
                    }

                    //This does not attempt to validade anything - i.e. it will match invalid emails
                    if (candidate.Slice(0, i - 1).IsEmailLocalPart() && candidate.Slice(i + 1).IsEmailDomainPart())
                    {
                        return true;
                    }

                    //FROM: https://en.wikipedia.org/wiki/Email_address
                    //addr-spec   =  local-part@domain
                    //The local-part of the email address may use any of these ASCII characters:
                    //      uppercase and lowercase Latin letters A to Z and a to z;
                    //      digits 0 to 9;
                    //      special characters !#$%&'*+-/=?^_`{|}~;
                    //      dot ., provided that it is not the first or last character unless quoted, and provided also that it does not appear consecutively unless quoted (e.g. John..Doe@example.com is not allowed but "John..Doe"@example.com is allowed);
                    //Exceptions:
                    //      space and "(),:;<>@[\] characters are allowed with restrictions (they are only allowed inside a quoted string, as described in the paragraph below, and in addition, a backslash or double-quote must be preceded by a backslash);
                    //      comments are allowed with parentheses at either end of the local-part; e.g. john.smith(comment)@example.com and (comment)john.smith@example.com are both equivalent to john.smith@example.com.


                    //The domain part is defined as follows:
                    //   The Internet standards (Request for Comments) for protocols mandate that component hostname labels may contain only:
                    //      the ASCII letters a through z (in a case-insensitive manner),
                    //      the digits 0 through 9,
                    //      the hyphen (-).
                    //   The original specification of hostnames in RFC 952, mandated that labels could not start with a digit or with a hyphen,
                    //   and must not end with a hyphen. However, a subsequent specification (RFC 1123) permitted hostname labels to start with digits.
                    //   No other symbols, punctuation characters, or blank spaces are permitted.
                }
                else if (candidate[i] == '.' && candidate[i - 1] != '.' && candidate[i + 1] != '.')
                {
                    dotCount++; //could be an IPv4 address
                }

                //TODO: how do we handle IPv6 addresses?

            }

            if (dotCount == 4) { return IsURLRegex(candidate); } //possibly an IP address

            if (dotCount > 0)
            {
                //Double check with a regex
                return IsURLRegex(candidate);
            }


            return false;

            //Test here if it's like a URL, email, etc...
            //URL_PATTERN = (
            //   r"^"
            //    # in order to support the prefix tokenization (see prefix test cases in test_urls).
            //    r"(?=[\w])"
            //    # protocol identifier
            //    r"(?:(?:https?|ftp|mailto)://)?"
            //    # user:pass authentication
            //    r"(?:\S+(?::\S*)?@)?"
            //    r"(?:"
            //    # IP address exclusion
            //    # private & local networks
            //    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
            //    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
            //    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
            //    # IP address dotted notation octets
            //    # excludes loopback network 0.0.0.0
            //    # excludes reserved space >= 224.0.0.0
            //    # excludes network & broadcast addresses
            //    # (first & last IP address of each class)
            //    # MH: Do we really need this? Seems excessive, and seems to have caused
            //    # Issue #957
            //    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
            //    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
            //    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
            //    r"|"
            //    # host name
            //    r"(?:(?:[a-z0-9\-]*)?[a-z0-9]+)"
            //    # domain name
            //    r"(?:\.(?:[a-z0-9\-])*[a-z0-9]+)*"
            //    # TLD identifier
            //    r"(?:\.(?:[a-z]{2,}))"
            //    r")"
            //    # port number
            //    r"(?::\d{2,5})?"
            //    # resource path
            //    r"(?:/\S*)?"
            //    # query parameters
            //    r"\??(:?\S*)?"
            //    # in order to support the suffix tokenization (see suffix test cases in test_urls),
            //    r"(?<=[\w/])"
            //    r"$"
            //).strip()

        }

        private static ConcurrentDictionary<int, string> ShapesCache = new ConcurrentDictionary<int, string>();

        private static readonly int _H_Base   = "shape".AsSpan().IgnoreCaseHash32();
        private static readonly int _H_Digit  = "shape_digit".AsSpan().IgnoreCaseHash32();
        private static readonly int _H_Lower  = "shape_lower".AsSpan().IgnoreCaseHash32();
        private static readonly int _H_Upper  = "shape_upper".AsSpan().IgnoreCaseHash32();
        private static readonly int _H_Punct  = "shape_puct".AsSpan().IgnoreCaseHash32();
        private static readonly int _H_Symbol = "shape_symbol".AsSpan().IgnoreCaseHash32();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string Shape(this ReadOnlySpan<char> token, bool compact)
        {
            if (token.Length == 0) return "";

            int hash = _H_Base;
            int prevType = _H_Base;
            for (int i = 0; i < token.Length; i++)
            {
                int type;
                if (char.IsLower(token[i])) { type = _H_Lower; }
                else if (char.IsUpper(token[i])) { type = _H_Upper; }
                else if (char.IsNumber(token[i])) { type = _H_Digit; }
                else if (char.IsPunctuation(token[i])) { type = _H_Punct; }
                else { type = _H_Symbol; }

                if (!compact || type != prevType)
                {
                    hash = Hashes.CombineWeak(hash, type);
                }
                prevType = type;
            }


            string shape;

            if(!ShapesCache.TryGetValue(hash, out shape))
            {
                var sb = new StringBuilder(token.Length);
                char prevchar = '\0', curchar;
                for (int i = 0; i < token.Length; i++)
                {
                    if (char.IsLower(token[i]))             { curchar = 'x'; }
                    else if (char.IsUpper(token[i]))        { curchar = 'X'; }
                    else if (char.IsNumber(token[i]))       { curchar = '9'; }
                    else if (char.IsPunctuation(token[i]))  { curchar = '.'; }
                    else { curchar = '#'; }

                    if (!compact || curchar != prevchar)
                    {
                        sb.Append(curchar);
                    }
                    prevchar = curchar;
                }
                shape = sb.ToString();
                ShapesCache[hash] = shape;
            }
            return shape;
        }

    }
}
