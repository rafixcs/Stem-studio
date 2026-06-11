# Stem Studio

Aplicação desktop (Windows e Linux) para:

- **Separar instrumentos** de uma faixa de música em 4 stems — vocais, bateria, baixo e outros — usando o modelo de IA **Demucs (htdemucs_6s)** — agora em 6 stems: vocais, bateria, baixo, **guitarra**, piano e outros;
- **Identificar a quantidade de guitarras** pela análise da imagem estéreo e dividir o stem de guitarra em uma faixa por guitarra (esquerda/centro/direita);
- **Controlar o volume** de cada instrumento individualmente (com mudo por stem);
- **Detectar automaticamente o BPM** da música (librosa);
- **Alterar o andamento (BPM)** sem alterar o pitch e com alta qualidade, usando o **Rubber Band**;
- **Metrônomo sincronizado com suporte a andamento variável**: a música é segmentada em trechos de tempo estável (via novidade do tempograma), cada trecho tem seu beat tracking e correção de oitava próprios, e os clicks seguem as batidas reais de cada seção;
- **Exportar a mixagem** resultante em WAV.

## Requisitos

- Python 3.10–3.12
- FFmpeg (para abrir MP3/M4A etc.)
- Rubber Band CLI (recomendado, para o time-stretch de alta qualidade)

### Linux (Ubuntu/Debian)

```bash
sudo apt install ffmpeg rubberband-cli libportaudio2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

1. Instale o Python em https://python.org (marque "Add to PATH").
2. Instale o FFmpeg (ex.: `winget install Gyan.FFmpeg`).
3. Baixe o Rubber Band CLI em https://breakfastquay.com/rubberband/ e coloque o `rubberband.exe` numa pasta do PATH.
4. No PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Se o `rubberband` não for encontrado, o app continua funcionando e usa o
> phase vocoder do librosa como fallback (qualidade um pouco menor).

### GPU (opcional, recomendado)

A separação no CPU funciona, mas é lenta (alguns minutos por música). Com uma
GPU NVIDIA, instale o PyTorch com CUDA antes do restante:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Executar

```bash
python -m stemstudio.main
```

## Interface e controles

- **Tema escuro estilo DAW** com cores por instrumento;
- **Forma de onda interativa**: clique para buscar uma posição; **arraste para criar um
  loop A–B** (perfeito para estudar um trecho — o loop é mantido proporcionalmente ao
  mudar o andamento);
- **Mixer com Solo (S) e Mudo (M)** por faixa + volume **Master**;
- **Atalhos**: Espaço (tocar/pausar), ←/→ (±5 s), L (loop), Home (início);
- **Arraste e solte** um arquivo de áudio direto na janela;
- **Exportar faixas…** salva cada stem como WAV individual (para DAW/looper).

## Desempenho

- **Cache automático** em `~/.stemstudio/cache`: a separação (Demucs) e a análise de
  batidas de cada música são reaproveitadas — reabrir a mesma faixa é instantâneo;
- **Time-stretch paralelo**: os stems são esticados simultaneamente (até 4 de uma vez);
- Player com **latência baixa** e barra de progresso real durante a separação.

## Uso

1. **Abrir música…** — o BPM é detectado automaticamente ao carregar.
2. **Separar instrumentos** — aguarde o Demucs processar (o modelo é baixado na primeira execução).
   Ao final, o app analisa a imagem estéreo do stem de guitarra, informa **quantas guitarras** foram
   identificadas e cria uma faixa no mixer para cada uma (ex.: "Guitarra 1 (esquerda)", "Guitarra 2 (direita)").

   > Como funciona: cada guitarra costuma ocupar uma posição de pan distinta na mixagem
   > (base dobrada esq./dir., solo ao centro). O app detecta essas posições por análise
   > espectral e separa por máscaras de pan. **Limitação**: guitarras mixadas exatamente
   > na mesma posição estéreo saem juntas numa única faixa — não há técnica atual que as distinga.
3. Ajuste os **sliders de volume** (0–150%) e os **mudos** de cada stem; a mixagem é em tempo real durante a reprodução.
4. Em **Andamento (BPM)**, digite o novo BPM e clique em **Aplicar novo BPM**. O stretch é sempre aplicado a partir dos stems originais (sem perda acumulada). **Restaurar original** volta ao andamento detectado.
5. **Metrônomo**: aparece como uma faixa no mixer (inicia **mudo**).
   Se a música tiver mudanças de andamento, o app detecta os trechos e mostra
   "BPM principal: X (varia em N trechos)" — passe o mouse sobre o rótulo para ver
   o BPM de cada trecho. A correção de erro de oitava é feita por trecho,
   automaticamente. Se ainda assim o metrônomo parecer na metade ou no dobro,
   use os botões **×2** / **÷2** ao lado do BPM (aplicam-se à faixa toda). Desmarque "Mudo" para
   ouvi-lo junto da música, com volume próprio. Os clicks seguem as batidas reais detectadas
   e são regenerados (não esticados) quando você muda o BPM — sempre secos e no tempo.
6. **Exportar mixagem…** salva o resultado (volumes + andamento) em WAV.

## Empacotar como executável (opcional)

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name StemStudio stemstudio/main.py
```

## Estrutura

```
stemstudio/
├── main.py          # ponto de entrada
├── main_window.py   # interface Qt (PySide6)
├── separator.py     # separação de stems (Demucs htdemucs_6s)
├── guitar_split.py  # contagem e separação de guitarras por pan estéreo
├── tempo.py         # detecção de BPM + time-stretch (Rubber Band / librosa)
├── metronome.py     # detecção de batidas + faixa de click sincronizada
└── player.py        # player com mixagem em tempo real (sounddevice)
```
# Stem-studio
