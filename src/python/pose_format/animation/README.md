# Animacija iš filtruotų pozų

Šiame kataloge yra iš `experiments/3248` perkelta ir katalogų apdorojimui
pritaikyta programa. Ji naudoja **Rocketbox `Bip01` skeletą**, ne bet kokį FBX.
Kūno ir plaštakų koordinatės interpretuojamos pagal šio projekto MediaPipe eksportą.

## Paleidimas

Aktyvioje pose projekto Python aplinkoje:

```powershell
animate_poses -i "H:\Science\ViSign\experiments" --recursive
```

Pagal nutylėjimą kuriami `.blend`, `.fbx` ir atliekama FBX eksporto patikra.
Peržiūros ir palyginimo video kuriami tik pridėjus **`--render`**:

```powershell
animate_poses -i "H:\Science\ViSign\experiments" --recursive --render
```

`--render` galima pridėti vėliau: jei animacija jau paruošta ir jos įvestys bei
nustatymai nepasikeitė, atliekamas tik renderinimas. Be šios vėliavėlės nereikia
FFmpeg ar originalaus video; kalibravimas, IK ir veido animacija vis tiek vykdomi.

Katalogai pagal nutylėjimą apdorojami rekursyviai, todėl `--recursive` galima
praleisti. Atrenkami tik `*_filtered.pose` failai. Vieną failą galima nurodyti su `-i`.
`--no-recursive` apriboja paiešką vienu katalogu.

Progresas rodomas automatiškai su `tqdm`: bendra `Animations` juosta skaičiuoja
pozų failus, o atskira juosta rodo esamo failo pavadinimą ir vykstantį etapą.
Animacijos kūrimui ir peržiūros renderinimui rodomi apdoroti / visi kadrai,
procentai, kadrų greitis ir likusio laiko įvertis. FBX patikros juosta skaičiuoja
tikrinamus mėginius, palyginimo video progresas gaunamas iš FFmpeg.
Duomenų paruošimas, modelio įkėlimas ir eksportas rodomi kaip atskiri veiksmai
be tariamo kadrų progreso. Kiekvieno etapo juosta pradedama nuo nulio;
jos likusio laiko įvertis taikomas tam etapui. Visi Blender / FFmpeg pranešimai
toliau saugomi žurnaluose. `--no-progress` išjungia progreso juostas ir nekeičia
generuojamų rezultatų ar jau užbaigtų darbų praleidimo.

Komanda įdiegiama kartu su paketu (`python -m pip install -e .` iš `pose/src/python`).
Alternatyva be naujo console entry point:

```powershell
python -m pose_format.animation.animate_poses -i "H:\Science\ViSign\experiments"
```

## Automatinis kalibravimas ir rankų IK

Numatytasis `--arm-solver ik` kiekvienam pozų failui automatiškai įvertina žmogaus
pečių plotį, žastų ir dilbių ilgius iš patikimų kadrų. Naudojama mediana su
išskirčių atmetimu; nereikia T pozos ar rankiniu būdu įvesti ūgio. Avataro matmenys
nuskaitomi iš etaloninio FBX kaulų. Klubai ir kojų taškai kalibravimui nenaudojami.

Riešų aukštis ir tarpusavio atstumas pirmiausia gaunami iš vaizdo taškų pečių
vidurio atžvilgiu. Aptiktas detalus plaštakos riešas naudojamas, jei nėra pernelyg
toli nuo kūno riešo. Pečių plotis suteikia bendrą vaizdo mastelį abiem rankoms;
kūno pasauliniai taškai ir žmogaus / avataro rankų ilgių santykis suteikia gylio
įvertį. Alkūnės taškas nurodo rankos lenkimo pusę.

Naudojamas analitinis dviejų kaulų IK: apskaičiuojama alkūnės padėtis ir kaulų
pasukimai, kurie iš karto įrašomi kaip įprasti rotacijų raktai. FBX nepriklauso
nuo Blender IK apribojimų ar pagalbinių objektų. Avataro kaulų ilgiai nekeičiami.
Nepasiekiami taikiniai apribojami pagal rankos pasiekiamumą. Taikiniai glodinami
prieš IK; išspręstos rankų rotacijos papildomai neglodinamos, kad riešai nenukryptų.
Delnų ir pirštų orientacijos toliau gaunamos iš esamos plaštakų animacijos.

Jei nėra bent 5 patikimų proporcijų įverčių, atitinkama ranka naudoja ankstesnį
rotacijų metodą; tai aiškiai įrašoma ataskaitoje. Trumpi stebėjimų tarpai
interpoliuojami, o ilgiems tarpams išlieka esama rankų atsarginė animacija.
Kalibravimas skirtas vienam žmogui ir nekintančiam kameros masteliui tame pačiame
faile; keičiantis žmogui ar kameros planui įrašą reikia suskaidyti į atkarpas.

`.animation/<pozos_vardas>/retarget_report.json` skyriai `arm_calibration` ir
`arm_ik` pateikia matmenis, panaudotų kadrų skaičius, atsarginio metodo priežastis,
apribotų taikinių skaičių ir skaičiavimo paklaidą. `ik_targets.npz` saugo riešų
taikinius FBX patikrai. `validation.json` papildomai tikrina riešų padėtis po
FBX pakartotinio importo. Maža ši paklaida patvirtina eksporto tikslumą, o ne
žmogaus tikrųjų 3D koordinačių tikslumą.

```powershell
animate_poses -i "H:\Science\ViSign\lrt_videos" --overwrite
```

Ankstesniam metodui palyginti naudokite `--arm-solver rotation`.
IK savaime nesprendžia pirštų kontakto ar plaštakų / rankų susikirtimo su kūnu.
Neapibrėžtas gylis ir prastai aptikti taškai vis dar gali sukelti neatitikimų.

## Veido mimika

Pagal nutylėjimą `--face-animation auto` ieško atitinkamo MediaPipe Tasks
failo: `lt_filtered.pose` → `lt.pose.extras.jsonl` (arba
`lt.mp4_filtered.pose` → `lt.mp4.pose.extras.jsonl`). Jei yra pačios filtruotos
pozos `lt_filtered.pose.extras.jsonl`, naudojamas jis.

```powershell
animate_poses -i "H:\Science\ViSign\lrt_videos" --recursive --overwrite
```

Rocketbox modelyje 51 MediaPipe koeficientas susiejamas su `AK_*` shape keys:
lūpos, žandikaulis, antakiai, skruostai, mirksėjimas ir akių kryptis.
`_neutral` nėra judesys; liežuvio `tongueOut` MediaPipe duomenyse nėra.
Kiti modelio mimikos rinkiniai (`AU_*`, visemos) lieka neutralūs, kad tas pats
judesys nebūtų pritaikytas kelis kartus. Koeficientai perkeliami be laikinio
glodinimo, kad neišnyktų trumpi mirksniai. Burnos atsivėrimui papildomai
taikomas toliau aprašytas kalibravimas.

Tikrinamas tikslus kadrų skaičius, indeksai ir laiko žymos. Apkarpant pozą
būtina apkarpyti ir sidecar, jo indeksus bei laiką pradėti nuo nulio.
Trumpi veido aptikimo tarpai iki 100 ms interpoliuojami, ilgesniuose veidas
neutralus. Klaidingas ar nesutampantis sidecar sustabdo to failo generavimą.
Jei sidecar nėra, kūno animacija veikia kaip anksčiau ir konsolė praneša
`missing_sidecar`. Senos CPU `.pose` veido taškų koordinatės savaime nėra
blendshape koeficientai: reikia MediaPipe Tasks sugeneruoto papildomo failo.

Mimikai išjungti: `--face-animation off`. Keičiant ar pridedant sidecar
rezultatas automatiškai perskaičiuojamas net ir be `--overwrite`.
Mimika išsaugoma tiek `.blend`, tiek `.fbx`. FBX patikra lygina visų susietų
kanalų reikšmes keliuose kadruose, įskaitant kiekvieno kanalo minimumą ir maksimumą.
Atitikmenys ir aptikimo statistika pateikiami `retarget_report.json`, patikros
rezultatai – `validation.json`. Tai mimikos perkėlimas; jis negarantuoja
gestų kalbos artikuliacijos tikslumo ir dar nėra MetaHuman veido integracija.

### Burnos atsivėrimo kalibravimas

Pagal nutylėjimą `--mouth-calibration auto` matuoja vidinių lūpų tarpo ir
burnos pločio santykį iš `.pose` veido taškų (13, 14, 61, 291). Matavimas
nepriklauso nuo vaizdo mastelio ar pasukimo vaizdo plokštumoje. Mažesnis nei
0,02 santykis laikomas užčiaupta burna. Labai mažas veidas, nepatikimi taškai,
stiprus galvos pasukimas arba trūkstama mimika palieka pradinius koeficientus.

Etaloniniam `Male_Adult_01` modeliui patikrinti keturi burnos geometrijos
taškai; profilis tikrinamas pagal tinklo topologiją. Kiekvienam kadrui
atsižvelgiama į visų pradinių mimikos koeficientų poveikį burnos pločiui ir
tarpui. Pirmiausia koreguojamas `jawOpen`; jei jo eigos nepakanka, koreguojami
`mouthLowerDownLeft/Right`, `mouthUpperUpLeft/Right` ir `mouthClose`.
Visi svoriai lieka 0–1 ribose. Mirksėjimas, antakiai, galva ir rankos nekeičiami.
Nenaudojamas viso klipo maksimumo normalizavimas – silpnas judesys netampa
maksimaliu išsižiojimu. Nepasiekiamas atsivėrimas lieka apribotas modelio eiga
ir pažymimas ataskaitoje. Kito topologijos modelio kalibravimas praleidžiamas.

Ankstesniam tiesioginiam perkėlimui palyginti:

```powershell
animate_poses -i "H:\Science\ViSign\experiments\facial_animation\man" --mouth-calibration off --overwrite
```

`retarget_report.json` → `face.mouth_calibration` pateikia panaudotų kadrų
skaičių, matavimo skirtumus prieš ir po korekcijos, užčiauptos burnos statistiką
ir pavyzdžių kadrus. FBX patikra papildomai matuoja pakartotinai importuotos
burnos geometriją tuose kadruose. Kalibravimui nereikia naujo video aptikimo,
jei jau turite `.pose` su veido taškais ir atitinkamą `.pose.extras.jsonl`.

Tai apytikslis matomo atsivėrimo pritaikymas. Jis negarantuoja viso lūpų
kontūro, dantų ar liežuvio padėties tikslumo; priklauso nuo aptiktų taškų.
Modelio matavimas atliekamas neutralioje galvos orientacijoje, todėl galvos
perspektyva peržiūroje gali pakeisti matomą santykį.

## Išvesties failai

Šalia `lt_filtered.pose` sukuriami:

- `lt_filtered_animated.blend`: redaguojama scena su animacija.
- `lt_filtered_animated.fbx`: FBX modelis su iškepta animacija.
- `lt_filtered_preview.mp4`: modelio animacijos peržiūra, be garso (su `--render`).
- `lt_filtered_comparison.mp4`: originalas kairėje, modelis dešinėje, be garso
  (su `--render`, jei rastas originalus video).

Tik su `--render` palyginimui ieškoma `lt.mp4` (ar kito palaikomo video formato) tame pačiame kataloge.
Palaikoma ir `lt.mp4_filtered.pose` -> `lt.mp4` pora. Originalaus video nesant,
animacija ir peržiūra sukuriamos, o palyginimas praleidžiamas. Esant keliems
vienodo pavadinimo originalams su skirtingais plėtiniais, failas pažymimas kaip
nepavykęs, kad nebūtų panaudotas neteisingas video.

Tarpiniai duomenys, aptikimo statistika, kaulų atitikmenys, FBX patikra ir žurnalai
saugomi `.animation/lt_filtered/`. Skirtingos pozos viename kataloge nesusiduria.
Peržiūra koduojama tiesiai į MP4; didelės PNG sekos neišsaugomos.

## Modelis ir įrankiai

Pirmiausia ieškoma artimiausio `reference_model/Export/Male_Adult_01_facial.fbx`
šalia pozos ar aukštesniuose kataloguose, taip pat projekto
`models/rocketbox/Male_Adult_01/Export/Male_Adult_01_facial.fbx`.
Modelį galima nurodyti aiškiai:

```powershell
animate_poses -i "H:\Science\ViSign\videos" --model "H:\Science\ViSign\models\rocketbox\Male_Adult_01\Export\Male_Adult_01_facial.fbx"
```

Blender ieškomas PATH ir Windows `Program Files/Blender Foundation` (pasirenkama
naujausia įdiegta versija). FFmpeg reikalingas palyginimo video. Galima nurodyti
`--blender "...\blender.exe"` ir `--ffmpeg "...\ffmpeg.exe"`.
MP4 generavimas pritaikytas Blender 4.4 ir 5.2 API.

Pagal nutylėjimą tekstūros bendros, nurodomos iš etaloninio modelio aplanko –
jų kopijos neįterpiamos į kiekvieną animaciją. Perkeliant rezultatus į kitą
kompiuterį naudokite `--embed-textures` (failai bus didesni).

## Pakartotinis generavimas

```powershell
animate_poses -i "H:\Science\ViSign\experiments" --overwrite
```

Be `--overwrite` praleidžiami tik užbaigti etapai su tais pačiais įvesties,
modelio, tekstūrų ir atitinkamo etapo programos failais bei parinktimis.
Animacija ir eksporto patikra žymimos `animation_completed.json`, renderinimas –
`render_completed.json` failu po `.animation/<pozos_vardas>/`.
Nepavykęs renderinimas išlaiko paruoštą animaciją. Dingus tik peržiūrai,
pasikeitus originaliam video arba `render_preview.py`, pakartojamas tik renderinimas.
`--overwrite` atnaujina animaciją ir, jei kartu pateiktas `--render`, jos video.
Be `--render` senos peržiūros neliečiamos ir gali nebeatitikti naujos animacijos;
jas atnaujins kitas paleidimas su `--render`.
Senieji bendri `completed.json` žymekliai nebenaudojami: po šio atnaujinimo
anksčiau apdorota animacija vieną kartą bus sugeneruota ir patikrinta iš naujo.
Vieno failo klaida nesustabdo kitų; komanda pabaigoje pateikia suvestinę ir grąžina
klaidos kodą, jei bent vienas darbas nepavyko. `--dry-run` tik išvardija įvestis.

## Ribos

### Piršto ir kitos rankos kontaktas (MĖGTI-D bandymas)

Jei šalia pozos yra `contact_profile.json`, `animate_poses` ir `run_pipeline.py`
automatiškai pritaiko jame aprašytą kontaktą. Pirmas palaikomas tipas:
`right_middle_to_left_hand_dorsum` — dešinės rankos vidurinio piršto galiukas
prie kairės plaštakos / riešo nugarinės pusės. Profilio pavyzdys:
`library/gestures/MĖGTI-D/default/contact_profile.json`.

Šiame pilote palietimo laikai **pažymėti pagal originalo peržiūrą**, o ne
automatiškai atpažinti. `strength` aprašo korekcijos įjungimą / išjungimą,
`clearance_cm` — prisilietimą ir atsitraukimą. Laikai sekundėmis tinka tiek
25, tiek 30 FPS. Originalaus video vardas ir SHA256 saugo nuo to paties profilio
atsitiktinio pritaikymo kito žmogaus įrašui. Vienas profilis skirtas vienam
originaliam video tame kataloge; pakeitus įrašą kontaktus reikia peržiūrėti.

Pagal avataro odos viršūnes nustatomi kontaktų žymekliai. Korekcija pasuka
vidurinį pirštą ties pagrindu, o dviejų kaulų IK pakoreguoja riešo padėtį,
išlaikydamas kaulų ilgius, plaštakos orientaciją ir pasyviąją ranką.
Ji įjungiama ir išjungiama tolygiai. Perkėlus į MetaHuman kontaktas dar kartą
sprendžiamas pagal konkretaus `MH_Signer` modelio geometriją. Rezultatas
iškepamas į įprastus animacijos raktus, todėl jį paveldi komponavimas.

`retarget_report.json` saugo profilį, žymeklius ir kiekvieno koreguoto kadro
paklaidas; `validation.json` patikrina eksportuoto FBX žymeklius.
Unreal importo ataskaitos `checks.hand_contacts` papildomai patikrina iškeptą
animaciją. Žymekliams naudojami visi pasirinktą odos viršūnę veikiantys kaulai
ir jų svoriai, įskaitant riešo korekcinius kaulus. Tai nėra pilna odos
kolizijų sistema: vaizdinė peržiūra išlieka būtina. Šis režimas nepatvirtina
gesto lingvistinio tikslumo ir neatkuria visų trūkstamų pirštų detalių.

Pakeitus profilį įprastas paleidimas atnaujina animaciją ir priklausomus
rezultatus. `--new-only` čia netinka, nes jis sąmoningai pasitiki jau užbaigtais
įrašais. Įrašai be profilio kontaktų korekcijos nenaudoja.

**Profilio 2 versija** palietimus saugo atskiruose `contact_intervals_seconds`
intervaluose. Tarp jų `release_trajectories[].up_hand_lengths` aprašo pakilimo
laiko kreivę: aukštis pateikiamas avataro pasyvios plaštakos riešo–vidurinio
piršto pagrindo ilgio dalimis. Pakilimas vyksta vertikaliai aukštyn; kontaktų
intervalais galiukas grąžinamas į plaštakos paviršių. Bendras `strength` valdo
korekcijos įėjimą / išėjimą, o ne nuolatinį piršto prispaudimą. `clearance_cm`
lieka priartėjimo ir galutinio atsitraukimo kreive už pažymėtos sekos ribų.

MĖGTI-D pirmas palietimas: 0,92–0,98 s; pakilimas: 0,98–1,12 s;
antras palietimas: 1,12–1,36 s. Pakilimo pikas 1,04 s siekia 0,35 plaštakos
ilgio. Tai vaizdiškai parinktas pirmo bandymo aukštis, ne išmatuotas 3D atstumas.
Piką ir visą kreivę galima keisti profilyje. Kiti pirštai, pasyvi ranka ir
judesiai už korekcijos intervalo išlaikomi. 1 versijos profiliai tebepalaikomi.

`taps` patikra Blender, pakartotinai įkeltame FBX ir Unreal tikrina atskirus
palietimus bei tikrą žymeklio pakilimą tarp jų. Susiliejus palietimams arba
pranykus pakilimui apdorojimas laikomas nepavykusiu.

### Pirštų formos perkėlimas plaštakos atžvilgiu

Šalia konkretaus `*_filtered.pose` galima padėti `handshape_profile.json`.
`animate_poses` ir `run_pipeline.py` jį automatiškai įtraukia į animavimo
užduotį. Pavyzdys: `library/gestures/AŠ/default/handshape_profile.json`.
Įrašai be šio failo animuojami kaip anksčiau.

Profilis saugo penkių pirštų trijų segmentų vienetines kryptis vietinėje
plaštakos sistemoje: X eina nuo riešo link vidurinio piršto pagrindo, Y —
link smiliaus pagrindo, Z — jų vektorinė sandauga. Donoro padėtis, mastelis
ir pasukimas pašalinami. Avataro plaštakos pasukimas ir kaulų ilgiai išlieka
savi; keičiami tik nurodytos rankos 15 pirštų kaulų pasukimai. Perkeliama
forma, o ne donoro rankos trajektorija. Kairės ir dešinės rankų veidrodinis
perkėlimas šiuo metu nenumatytas — rinktis tą pačią pusę.

`strength` yra sekundžių ir stiprumo (0–1) poros. Už intervalo korekcijos
nėra, o įėjimas ir išėjimas glotninami. Pilno stiprumo metu laikoma viena
donoro forma; tai tinka pasirinktam statinės formos intervalui, tačiau
netinka visam gestui, kuriame pati pirštų forma keičiasi. Po šio žingsnio,
jeigu yra kontaktų profilis, kontaktų korekcija turi pirmenybę.

`handshape_profile.extract_template` sudaro formą iš peržiūrėtų, aptiktų
donoro kadrų paruoštame NPZ. AŠ bandymo parengimas aprašytas
`experiments/as-handshape/prepare_experiment.py`: naudojama sakinio video
1,16–1,32 s atkarpa, o AŠ koreguojamas 0,28–1,72 s intervale.
`provenance` registruoja donorą, kadrus ir matavimų sklaidą.

Originalūs `.pose` ir pasitikėjimo įverčiai nekeičiami. Profilis susietas su
konkrečiais video ir pozos SHA256; pakeitus šiuos šaltinius būtina peržiūrėti
profilį. Profilio pakeitimai keičia animacijos kontrolinį atspaudą, todėl
įprastas paleidimas pergeneruoja priklausomus rezultatus (`--new-only`
šioms korekcijoms netinka). Profilio kopija saugoma animacijos užduotyje ir
`retarget_report.json`, taigi patenka ir į bibliotekos `animation_data`.

Animavimo patikra visuose kadruose tikrina, kad nepakeisti kitų kaulų
vietiniai pasukimai bei pirštai už intervalo, ir kad pilnu stiprumu pirštų
segmentai atitinka užduotas kryptis. Papildomai tikrinamas FBX importas.
Tai rekonstruota forma iš vienos kameros, ne naujai išmatuoti paslėpti
pirštai. Ji netaiso klaidingos pačios plaštakos orientacijos ar kontakto
su kūnu, todėl būtina vaizdinė peržiūra.

### Bendros rekonstrukcijos ribos

Animuojami žastai, dilbiai, plaštakos, pirštai, galva ir du viršutiniai stuburo
kaulai (39 kaulai iš viso). Galvos pasukimas, linktelėjimas ir šoninis palenkimas
gaunami iš standaus viršutinės veido dalies taškų sutapdinimo; lūpų ir žandikaulio
taškai nenaudojami. Veido mimika atskirai gaunama iš Tasks blendshape koeficientų.

Liemens pasisukimas ir šoninis pasvirimas skaičiuojami tik iš pečių linijos.
Nematomų ar filtruotų klubų koordinatės nenaudojamos. Lenkimasis pirmyn/atgal ir
visos figūros poslinkis neatkuriami, nes vien pečių linija jų nenusako.
Pirmas iki 0,4 s ilgio vientisas aptikimo intervalas laikomas neutralia padėtimi;
taip pašalinamas pastovus kameros/žmogaus pradinis pasvirimas. Jei įrašas prasideda
pasukta galva ar kūnu, ši pradinė poza taip pat bus laikoma neutralia.

Galvos tikslinė orientacija neprideda liemens pasukimo antrą kartą. Nejudinamas
apatinis stuburo kaulas, prie kurio Rocketbox skelete prijungtos kojos; dubuo ir
kojos išlieka stabilūs. Kaklas ir raktikauliai paveldi viršutinio liemens judesį.
Nepatikimuose galvos/liemens kadruose laikomas
paskutinis vietinis pasukimas. Kai nėra veido komponento ar tinkamų taškų,
galva tiesiog paveldi liemens judesį.

Trumpi aptikimo tarpai interpoliuojami,
ilgesniuose laikomi paskutiniai vietiniai plaštakų/pirštų pasukimai; nepatikimos
rankos nuleidžiamos į neutralią pozą. Tai apytikslė vienos kameros rekonstrukcija.

Kiekvieno FBX kaulų transformacijos keliuose kadruose palyginamos su Blender
scena, tikrinama trukmė, apatinės kūno dalies stabilumas pasaulio koordinatėse
ir neanimuotų kaulų vietinių pasukimų stabilumas. Ši patikra nepatvirtina
gestų kalbos tikslumo ar MetaHuman retargetinimo.
