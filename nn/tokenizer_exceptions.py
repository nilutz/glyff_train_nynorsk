from spacy.lang.tokenizer_exceptions import BASE_EXCEPTIONS
from spacy.symbols import ORTH, NORM
from spacy.util import update_exc


_exc = {}


for exc_data in [
    {ORTH: "jan.", NORM: "januar"},
    {ORTH: "feb.", NORM: "februar"},
    {ORTH: "mar.", NORM: "mars"},
    {ORTH: "apr.", NORM: "april"},
    {ORTH: "jun.", NORM: "juni"},
    {ORTH: "jul.", NORM: "juli"},
    {ORTH: "aug.", NORM: "august"},
    {ORTH: "sep.", NORM: "september"},
    {ORTH: "okt.", NORM: "oktober"},
    {ORTH: "nov.", NORM: "november"},
    {ORTH: "des.", NORM: "desember"},
]:
    _exc[exc_data[ORTH]] = [exc_data]



for orth in [
    "a.D.",
    "Ap.",
    "Aq.",
    "Ca.",
    "Chr.",
    "Co.",
    "Co.",
    "Dr.",
    "F.eks.",
    "Fr.p.",
    "Frp.",
    "Grl.",
    "Kr.",
    "Kr.F.",
    "Kr.F.s",
    "Mr.",
    "Mrs.",
    "Pb.",
    "Pr.",
    "Sp.",
    "Sp.",
    "St.",
    "a.m.",
    "ad.",
    "adm.dir.",
    "andelsnr",
    "b.c.",
    "bl.a.",
    "bla.",
    "bm.",
    "bnr.",
    "bto.",
    "c.c.",
    "ca.",
    "cand.mag.",
    "co.",
    "d.d.",
    "d.m.",
    "d.y.",
    "dept.",
    "dr.",
    "dr.med.",
    "dr.philos.",
    "dr.psychol.",
    "dvs.",
    "e.Kr.",
    "e.l.",
    "eg.",
    "ekskl.",
    "el.",
    "et.",
    "etc.",
    "etg.",
    "ev.",
    "evt.",
    "f.",
    "f.Kr.",
    "f.eks.",
    "f.o.m.",
    "fhv.",
    "fk.",
    "foreg.",
    "fork.",
    "fv.",
    "fvt.",
    "g.",
    "gl.",
    "gno.",
    "gnr.",
    "grl.",
    "gt.",
    "h.r.adv.",
    "hhv.",
    "hoh.",
    "hr.",
    "ifb.",
    "ifm.",
    "iht.",
    "inkl.",
    "istf.",
    "jf.",
    "jr.",
    "jun.",
    "juris.",
    "kfr.",
    "kgl.",
    "kgl.res.",
    "kl.",
    "komm.",
    "kr.",
    "kst.",
    "lat.",
    "lø.",
    "m.a.o.",
    "m.fl.",
    "m.m.",
    "m.v.",
    "ma.",
    "mag.art.",
    "md.",
    "mfl.",
    "mht.",
    "mill.",
    "min.",
    "mnd.",
    "moh.",
    "mrd.",
    "muh.",
    "mv.",
    "mva.",
    "n.å.",
    "ndf.",
    "no.",
    "nov.",
    "nr.",
    "nto.",
    "nyno.",
    "o.a.",
    "o.l.",
    "off.",
    "ofl.",
    "okt.",
    "on.",
    "op.",
    "org.",
    "osv.",
    "ovf.",
    "p.",
    "p.a.",
    "p.g.a.",
    "p.m.",
    "p.t.",
    "pga.",
    "ph.d.",
    "pkt.",
    "pr.",
    "pst.",
    "pt.",
    "red.anm.",
    "ref.",
    "res.",
    "res.kap.",
    "resp.",
    "rv.",
    "s.",
    "s.d.",
    "s.k.",
    "s.k.",
    "s.u.",
    "s.å.",
    "sen.",
    "sep.",
    "siviling.",
    "sms.",
    "snr.",
    "spm.",
    "sr.",
    "sst.",
    "st.",
    "st.meld.",
    "st.prp.",
    "stip.",
    "stk.",
    "stud.",
    "sv.",
    "såk.",
    "sø.",
    "t.h.",
    "t.o.m.",
    "t.v.",
    "temp.",
    "ti.",
    "tils.",
    "tilsv.",
    "tl;dr",
    "tlf.",
    "to.",
    "ult.",
    "utg.",
    "v.",
    "vedk.",
    "vedr.",
    "vg.",
    "vgs.",
    "vha.",
    "vit.ass.",
    "vn.",
    "vol.",
    "vs.",
    "vsa.",
    "©NTB",
    "årg.",
    "årh.",
    "§§",
]:
    _exc[orth] = [{ORTH: orth}]


TOKENIZER_EXCEPTIONS = update_exc(BASE_EXCEPTIONS, _exc)