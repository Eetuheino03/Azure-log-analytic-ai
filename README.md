# Azure Log Analytic AI - koulutusdatan valmistelu

Tämä projekti hakee Azure Sentinel -harjoitusaineiston GitHubista ja muuntaa sen luokitelluksi koulutusdataksi.

## Suositus aloitukseen
Käytä ensin binary-datasettia:
- `data/processed/training_dataset_binary.csv`
- `data/processed/training_dataset_binary.jsonl`

Nykyisellä aineistolla (2026-02-23 ajon tulos):
- `suspicious`: 23,945
- `normal`: 14,752
- `unknown`: poistettu binary-versiosta

## Mallipohja (logem)
Koulutukseen käytä Transformers-mallia:
- `HassanShehata/logem` (Qwen3-0.6B fine-tune)

Huomio:
- `LLMSIEM/logem` palauttaa API:ssa `401 Unauthorized` (todennäköisesti private/gated)
- `mradermacher/logem-GGUF` on GGUF-quant inferenssiä varten, ei LoRA-fine-tuningiin

## Lähdedata
- Repo: `Azure/Azure-Sentinel`
- Polku: `Solutions/Training/Azure-Sentinel-Training-Lab/Artifacts/Telemetry`
- Muoto: CSV-logit

## Mitä skripti tekee
Skripti `scripts/prepare_training_data.py`:
1. Lataa CSV-tiedostot GitHubista kansioon `data/raw/`
2. Normalisoi rivit tekstiksi (`text`-kenttä)
3. Luokittelee rivit (`log_type`, `risk_label`, `target`)
4. Jakaa datan: `train` 80%, `val` 10%, `test` 10%
5. Tuottaa 3 datasettiä:
   - `original` (tarkka moniluokka)
   - `binary` (`normal` vs `suspicious`, `unknown` poistettu)
   - `medium` (SOC-tyyppiset karkeammat luokat)

## Datan valmistelu
```powershell
python scripts\prepare_training_data.py
```

## LoRA-koulutus (binary)
Asenna riippuvuudet:
```powershell
pip install -r requirements.txt
```

Jos saat virheen `model type 'qwen3' ... Transformers does not recognize this architecture`, päivitä Transformers uusimpaan:
```powershell
pip install --upgrade git+https://github.com/huggingface/transformers.git
```

Käynnistä koulutus:
```powershell
python scripts\train_logem_binary_lora.py \
  --model_name HassanShehata/logem \
  --dataset_csv data/processed/training_dataset_binary.csv \
  --output_dir outputs/logem-binary-lora
```

Skripti tekee oletuksena:
- train-splitin tasapainotuksen (`--balance_train`)
- 4-bit latauksen GPU:lla (`--use_4bit`)
- CPU-optimoidun profiilin `--profile cpu_fast`
- tokenisoinnin cachen (`--cache_tokenized`)
- run lockin (`--run_lock`) estämään vahingossa rinnakkaiset ajot samaan output-kansioon

### CPU-optimoidut profiilit
- `cpu_fast` (oletus): nopein baseline CPU:lla
- `cpu_balanced`: kompromissi nopeuden ja laadun välillä
- `cpu_quality`: hitaampi, tarkempi

Esimerkki CPU-ajoon:
```powershell
python scripts\train_logem_binary_lora.py \
  --profile cpu_fast \
  --cpu_threads 24 \
  --interop_threads 1 \
  --tokenize_workers 12 \
  --model_name HassanShehata/logem \
  --dataset_csv data/processed/training_dataset_binary.csv \
  --output_dir outputs/logem-binary-lora
```

Debug/smoke ajo:
```powershell
python scripts\train_logem_binary_lora.py --max_train_rows 1024 --eval_samples_final 100
```

Poista tasapainotus:
```powershell
python scripts\train_logem_binary_lora.py --no_balance_train
```

## Ulostulot
- `data/processed/training_dataset.csv`
- `data/processed/training_dataset.jsonl`
- `data/processed/summary.json`
- `data/processed/training_dataset_binary.csv`
- `data/processed/training_dataset_binary.jsonl`
- `data/processed/summary_binary.json`
- `data/processed/training_dataset_medium.csv`
- `data/processed/training_dataset_medium.jsonl`
- `data/processed/summary_medium.json`

Koulutuksen jälkeen:
- `outputs/logem-binary-lora/` (LoRA adapter + tokenizer)

## Datasetin skeema
Peruskentät:
- `id`
- `source_file`
- `log_type`
- `risk_label`
- `target`
- `split`
- `text`

Lisäkentät:
- Binary: `binary_label`, `y` (0/1)
- Medium: `medium_label`
