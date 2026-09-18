# Animacija iš filtruotų pozų

Šiame kataloge yra iš `experiments/3248` perkelta ir katalogų apdorojimui
pritaikyta programa. Ji naudoja **Rocketbox `Bip01` skeletą**, ne bet kokį FBX.
Kūno ir plaštakų koordinatės interpretuojamos pagal šio projekto MediaPipe eksportą.

## Paleidimas

Aktyvioje pose projekto Python aplinkoje:

```powershell
animate_poses -i "H:\Science\ViSign\experiments" --recursive
```

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

## Rezultatai

Šalia `lt_filtered.pose` sukuriami:

- `lt_filtered_animated.blend`: redaguojama scena su animacija.
- `lt_filtered_animated.fbx`: FBX modelis su iškepta animacija.
- `lt_filtered_preview.mp4`: modelio animacijos peržiūra, be garso.
- `lt_filtered_comparison.mp4`: originalas kairėje, modelis dešinėje, be garso.

Palyginimui ieškoma `lt.mp4` (ar kito palaikomo video formato) tame pačiame kataloge.
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

Be `--overwrite` praleidžiami tik užbaigti rezultatai su tais pačiais įvesties,
modelio, tekstūrų ir programos failais bei parinktimis. Pasikeitus jiems arba
dingus rezultatui, generuojama iš naujo. Nepavykę darbai nepažymimi kaip baigti.
Vieno failo klaida nesustabdo kitų; komanda pabaigoje pateikia suvestinę ir grąžina
klaidos kodą, jei bent vienas darbas nepavyko. `--dry-run` tik išvardija įvestis.

## Ribos

Animuojami žastai, dilbiai, plaštakos, pirštai, galva ir du viršutiniai stuburo
kaulai (39 kaulai iš viso). Galvos pasukimas, linktelėjimas ir šoninis palenkimas
gaunami iš standaus viršutinės veido dalies taškų sutapdinimo; lūpų ir žandikaulio
taškai nenaudojami. Tai galvos orientacija, o ne veido mimikos animacija.

Liemens pasisukimas ir šoninis pasvirimas skaičiuojami tik iš pečių linijos.
Nematomų ar filtruotų klubų koordinatės nenaudojamos. Lenkimasis pirmyn/atgal ir
visos figūros poslinkis neatkuriami, nes vien pečių linija jų nenusako.
Pirmas iki 0,4 s ilgio vientisas aptikimo intervalas laikomas neutralia padėtimi;
taip pašalinamas pastovus kameros/žmogaus pradinis pasvirimas. Jei įrašas prasideda
pasukta galva ar kūnu, ši pradinė poza taip pat bus laikoma neutralia.

Galvos tikslinė orientacija neprideda liemens pasukimo antrą kartą. Nejudinamas
apatinis stuburo kaulas, prie kurio Rocketbox skelete prijungtos kojos; dubuo ir
kojos išlieka stabilūs. Kaklas ir raktikauliai paveldi viršutinio liemens judesį.
Mimika kol kas neanimuojama. Nepatikimuose galvos/liemens kadruose laikomas
paskutinis vietinis pasukimas. Kai nėra veido komponento ar tinkamų taškų,
galva tiesiog paveldi liemens judesį.

Trumpi aptikimo tarpai interpoliuojami,
ilgesniuose laikomi paskutiniai vietiniai plaštakų/pirštų pasukimai; nepatikimos
rankos nuleidžiamos į neutralią pozą. Tai apytikslė vienos kameros rekonstrukcija.

Kiekvieno FBX kaulų transformacijos keliuose kadruose palyginamos su Blender
scena, tikrinama trukmė, apatinės kūno dalies stabilumas pasaulio koordinatėse
ir neanimuotų kaulų vietinių pasukimų stabilumas. Ši patikra nepatvirtina
gestų kalbos tikslumo ar MetaHuman retargetinimo.
