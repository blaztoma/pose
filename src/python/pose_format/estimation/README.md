# Pozų aptikimo realizacijos

Vienas `pose_format` paketas, du pasirenkami aptikimo metodai:

| `--backend` | Aplinka | Paskirtis |
| --- | --- | --- |
| `mediapipe-legacy` (numatytasis) | `.venv`, MediaPipe 0.10.21 | Esamas `mp.solutions.holistic`, CPU |
| `mediapipe-tasks` | `.venv-tasks`, MediaPipe 1.0.1 | Holistic Tasks VIDEO, veidas, kūnas, rankos ir papildomi mimikos duomenys |

`base.py` apibrėžia bendrą sąsają ir pasirinkimą. Kiekviena realizacija yra
atskirame modulyje, o MediaPipe importuojamas tik pasirinkus realizaciją.
`holistic_schema.json` eksportuotas iš esamo
`utils.holistic.holistic_components(additional_face_points=10)`:
taškų vardai, tvarka, jungtys ir spalvos lieka tokie patys.

## Aplinkos ir modelis

Šiame kompiuteryje abi aplinkos jau paruoštos. Naujos aplinkos atkūrimas:

```powershell
cd H:\Science\ViSign\pose\src\python
py -3.12 -m venv .venv-tasks
.\.venv-tasks\Scripts\python.exe -m pip install -r requirements-tasks.txt
```

Nediekite abiejų MediaPipe versijų į tą pačią aplinką.
`requirements-tasks.txt` fiksuoja išbandytą MediaPipe 1.0.1.
MediaPipe pats įdiegia `opencv-contrib-python`; papildomo `opencv-python` nereikia.

Prieš paleidžiant Tasks tikrinama įdiegtos MediaPipe versija (`>=1.0.1,<2`).
Jei komanda paleista iš senos `.venv`, gaunamas aiškus pranešimas su aktyvaus
Python keliu ir nurodymu naudoti `.venv-tasks`. Tai svarbu: MediaPipe 0.10.21
senieji Tasks bindings gali nutraukti procesą su `The packet is empty`, kai
trūksta dalies taškų. Tokiu atveju `BrokenProcessPool` yra proceso griūties
pasekmė; mažesnis `--num-workers` netinkamos versijos problemos neišsprendžia.
Ši klaida atkurta su `HyperDeck_0003.mp4` pradžia senoje aplinkoje; naujoje
aplinkoje pirmi 30 to paties video kadrų apdoroti sėkmingai.

Oficialus modelis jau atsisiųstas į
`H:\Science\ViSign\models\mediapipe\holistic_landmarker.task`.
Jis automatiškai ieškomas dabartinio katalogo ir paketo katalogo protėviuose,
`models/mediapipe/holistic_landmarker.task` vietoje. Kitai vietai naudokite
`--model "kelias\holistic_landmarker.task"`.
Programa modelių automatiškai nesisiunčia.

[Modelio dokumentacija](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker)
ir [oficialus atsisiuntimas](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task).

## Naujas metodas

Galima naudoti konkrečios aplinkos vykdomąjį failą, nekeičiant aktyvios aplinkos:

```powershell
H:\Science\ViSign\pose\src\python\.venv-tasks\Scripts\videos_to_poses.exe `
  --directory "H:\Science\ViSign\videos" --recursive `
  --backend mediapipe-tasks --device cpu --num-workers 2 `
  --output-directory "H:\Science\ViSign\videos_tasks"
```

Išlaikoma pakatalogių struktūra, ankstesnės pozos `videos` kataloge neliečiamos.
Jei naujame rezultatų kataloge `.pose` jau yra, jis praleidžiamas.
Pakartotiniam generavimui pasirinkite kitą rezultatų katalogą.
Be `--output-directory` pozos rašomos greta originalų, kaip ir anksčiau.

Pridėjus `--filter`, kartu išsaugoma ir `*_filtered.pose` kopija: kelių,
čiurnų, kulnų ir pėdų taškų patikimumas nustatomas į nulį, taškai užmaskuojami.
Klubai paliekami. Originali poza bei veido `.extras.jsonl` duomenys išsaugomi.

```powershell
videos_to_poses --directory "H:\Science\ViSign\videos" --recursive `
  --backend mediapipe-tasks --device cpu --filter
```

Jei originalus `.pose` jau yra, sukuriamas tik trūkstamas `_filtered.pose`,
nekartojant video apdorojimo ir neįkeliant aptikimo modelio. Jei abu failai yra,
įrašas praleidžiamas. Nauja poza filtruojama tiesiai iš atminties po originalo
išsaugojimo. Veikia ir su `--output-directory`, `--keep-video-suffixes` bei
keliais procesais. Be `--filter` filtruota kopija nekuriama.
Atskira `filter_poses` komanda lieka prieinama, taip pat jos `--include-hips` parinktis.

Įvesties kataloge ieškoma visų palaikomų video, įskaitant jame esančias
vizualizacijų kopijas, todėl dideliam paleidimui naudokite originalių video katalogą.

Vienam video:

```powershell
H:\Science\ViSign\pose\src\python\.venv-tasks\Scripts\video_to_pose.exe `
  -i "H:\Science\ViSign\experiments\3248\lt.mp4" `
  -o "H:\Science\ViSign\experiments\3248\mediapipe_tasks\lt.pose" `
  --backend mediapipe-tasks --device cpu
```

Vieno failo komanda perrašo nurodytą išvestį tik sėkmingai baigusi skaičiavimą.
Filtravimas, vizualizavimas ir animacija naudoja esamas komandas.
Overlay gali rasti originalų video pagal `.pose.meta.json`, net kai rezultatai
kitame kataloge; tai veikia ir `_filtered.pose` failams.
Animacijos palyginimo video kūrėjas vis dar ieško originalo greta pozos.

Norint aktyvuoti aplinką terminale:

```powershell
& "H:\Science\ViSign\pose\src\python\.venv-tasks\Scripts\Activate.ps1"
```

## Esamas CPU metodas

Ankstesnės komandos toliau veikia iš `.venv`. Pavyzdžiui:

```powershell
H:\Science\ViSign\pose\src\python\.venv\Scripts\videos_to_poses.exe `
  --directory "H:\Science\ViSign\videos" --recursive `
  --additional-config "model_complexity=2,refine_face_landmarks=true"
```

Numatytieji pasirinkimai yra `--backend mediapipe-legacy --device cpu`.
Senas `--format mediapipe` taip pat išliko.

## GPU ir parametrai

Šiame Windows kompiuteryje GPU **neveikia** su oficialiu MediaPipe 1.0.1
paketu: tiesioginis bandymas su GPU delegate grąžino
`ImageCloneCalculator: GPU processing is disabled in build flags`.
Todėl `--device gpu` Windows sistemoje atmetamas prieš video apdorojimą.
Vien CUDA įdiegimas šio paketo nepadarys GPU paketu.

Linux sistemoje `--backend mediapipe-tasks --device gpu` perduoda tikrą GPU
delegate į MediaPipe. Tam reikia jį palaikančio paketo ir veikiančios grafikos
aplinkos. Šiame darbe Linux / WSL GPU nebuvo išbandytas.
Nepavykus inicializacijai programa nesikartoja su CPU.

Tasks naudoja modelių rinkinį: seno `model_complexity=2`,
`refine_face_landmarks` ir `smooth_landmarks` parametrų čia nėra.
Veido 478 taškai grąžinami tiesiogiai. Leistini `--additional-config` parametrai:

- `min_face_detection_confidence`, `min_face_suppression_threshold`, `min_face_landmarks_confidence`;
- `min_pose_detection_confidence`, `min_pose_suppression_threshold`, `min_pose_landmarks_confidence`;
- `min_hand_landmarks_confidence`;
- `output_face_blendshapes=true` (numatyta).

Tasks apdoroja vieno video kadrus iš eilės ir išlaiko sekimo būseną.
Vieno video `--workers` turi būti 1; atskirus video galima lygiagretinti
naudojant `videos_to_poses --num-workers`. Daugiau procesų naudoja daugiau RAM.

Abu metodai automatiškai rodo `tqdm` progresą kiekvienam video: jo pavadinimą,
apdorotus / visus kadrus, procentus, skaičiavimo greitį ir likusio laiko įvertį.
Katalogo apdorojimas papildomai rodo bendrą video progresą. Su keliais procesais
kiekvienas tuo metu apdorojamas video turi atskirą juostą, o terminalą atnaujina
tik pagrindinis procesas. Kadras skaičiuojamas atlikus pozos aptikimą.
Jei video metaduomenyse nėra kadrų skaičiaus, rodomas apdorotų kadrų skaičius
ir greitis be procentų / likusio laiko. `--no-progress` išjungia juostas
abiejose komandose (`video_to_pose` ir `videos_to_poses`).

## Išvestis ir koordinačių susitarimai

Kiekvienam Tasks video sukuriama:

- `vardas.pose`: 33 kūno + 478 veido + 21 kairės rankos + 21 dešinės rankos
  + 33 kūno pasauliniai taškai. Nerasti taškai turi nulinį patikimumą.
- `vardas.pose.extras.jsonl`: po eilutę kadrui su `frame_index`, `timestamp_ms`,
  `face_blendshapes`, `left_hand_world_landmarks`, `right_hand_world_landmarks`.
  Nerasta ranka yra tuščias sąrašas, nerastas veidas – tuščias mimikos žodynas.
- `vardas.pose.meta.json`: metodas, įrenginys, MediaPipe versija, originalas,
  FPS, modelio kelias ir SHA256, konfigūracija, koordinačių paaiškinimai.
  Legacy metodas taip pat rašo metaduomenis.

`.pose` tiksliai išlaiko seną koordinačių sutartį: visų komponentų x dauginamas
iš pločio, y iš aukščio, z nekeičiamas. Tai taikoma ir `POSE_WORLD_LANDMARKS`;
esamas animacijos kodas šį mastelį panaikina. Papildomo JSONL rankų pasaulinės
koordinatės yra tiesioginiai MediaPipe duomenys metrais, su atskira kiekvienos
rankos lokalia pradžia. Jų negalima tiesiog sudėti su kūno koordinatėmis.
Veido / rankų `.pose` patikimumas 1 reiškia, kad komponentas aptiktas;
tai nėra pamatuotas kiekvieno taško tikslumas. Kūnui naudojamas `visibility`.

JSONL mimikos koeficientai išsaugomi ateities veido animacijai;
esamas `animate_poses` jų dar neperkelia į FBX / MetaHuman.
Laiko žymės skaičiuojamos iš kadro numerio ir FPS (pastovaus FPS laiko skalė,
kaip esamame `.pose` formate), ne iš kintamo FPS video originalių PTS.

## Patikra 2026-09-17

Eksperimentas `experiments/3248/mediapipe_tasks`:

- Apdoroti visi 145 kadrai, 25 FPS. Sukurta poza, papildomi duomenys ir overlay.
- 52 veido mimikos koeficientai kiekviename šio klipo kadre.
- Rankų aptikimas: kairė 66/145, dešinė 39/145;
  anksčiau išsaugotoje `lt.pose` atitinkamai 63/145 ir 33/145.
  Tai aptikimo dažnis viename klipe, ne gestų tikslumo įvertinimas.
- `filter_poses` ir `animate_poses` sukūrė FBX, Blender sceną ir palyginimo video.
  FBX pakartotinio importo patikra praėjo; apatinė kūno dalis išliko statiška.
- Legacy adapterio koordinatės ir patikimumai tiksliai sutapo su tiesioginiu
  seno `load_holistic` rezultatu. Tasks antraštė atitinka seną 478 veido taškų schemą.
- Išbandytas rekursyvus dviejų video apdorojimas su dviem procesais,
  atskiru rezultatų katalogu ir jau sukurtų failų praleidimu.

Automatiniai testai iš `pose/src/python`:

```powershell
.\.venv-tasks\Scripts\python.exe -m unittest tests.estimation_test tests.animation_test tests.motion_test
.\.venv\Scripts\python.exe -m unittest tests.estimation_test tests.animation_test tests.motion_test
```
